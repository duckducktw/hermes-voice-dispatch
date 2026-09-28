"""openWakeWord 的 Hermes 喚醒詞包裝。

daemon 的音訊串流是 16 kHz / mono / int16，每次 1024 samples；openWakeWord
則以 80 ms（1280 samples）為最佳推論單位。本模組負責重新切塊、雙層命中判定
（強命中＝單幀過高門檻；弱命中＝短視窗內多幀過低門檻）與重置，讓呼叫端只要
一直餵 daemon 的原始 block 即可。
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz


class OwwUnavailable(RuntimeError):
    """openWakeWord 模型或執行期不可用。"""


class OwwSpotter:
    """以自訂 ONNX 模型偵測喚醒詞。

    **判定規則（2026-09-27 第四輪修訂）＝ 雙層**：

        強命中：任一幀 `score >= threshold`（單幀就成立）
        弱命中：最近 `window_frames` 幀內有 >= `relaxed_hits` 幀 `score >= relaxed_threshold`

    為什麼要雙層：這顆 model 的分數是**單幀尖峰**（真命中會衝到 0.9x，但常常只有
    一兩幀；用來確認的「連續 N 幀」因此會殺掉正確命中）。可是純單幀低門檻又太鬆
    （雜訊/外洩語音只要一幀過 0.45 就醒）。所以：
      - 真命中 → 通常直接走「強命中」（0.9x）秒醒；
      - 稍弱的真命中（0.7x、但前後幀也偏高）→ 走「弱命中」仍會醒；
      - 單一雜訊尖峰（只有一幀 0.5~0.8、鄰居都很低）→ **不再觸發**。

    `relaxed_threshold` 於 2026-09-27 由 0.40 提到 **0.60**：使用者「看影片什麼都
    沒說就被回」，log 顯示影片聲常態落在 0.40~0.69、真喊 0.85~0.97 —— 0.60 切在
    那道斷層上。

    命中後立即 reset，避免同一句話被重複喚醒。
    """

    def __init__(
        self,
        model_path: str,
        threshold: float,
        vad_threshold: float = 0.0,
        logger=None,
        relaxed_threshold: float = 0.60,
        window_frames: int = 6,
        relaxed_hits: int = 2,
        min_ac_rms: float = 0.0012,
    ):
        self._log = logger or logging.getLogger(__name__)
        self.model_path = os.path.expanduser(model_path)
        self.threshold = float(threshold)
        self.relaxed_threshold = float(relaxed_threshold)
        self.window_frames = max(1, int(window_frames))
        self.relaxed_hits = max(1, int(relaxed_hits))
        # 2026-09-28：去 DC 後的 AC-RMS 靜音閘。低於此值＝麥克風沒訊號（死訊號／純 DC），
        # 直接記 0 分不送模型。實測死訊號時 AC-RMS ≈ 0.0001~0.0003、
        # 真喊「hey hermes」時 >= 0.01，中間有 30 倍以上的餘裕。0 = 停用。
        self.min_ac_rms = float(min_ac_rms)
        self.vad_threshold = float(vad_threshold)
        self._buffer = np.zeros(0, dtype=np.int16)
        self._recent: deque = deque(maxlen=self.window_frames)
        self.consecutive_frames = 0
        self.latest_score = 0.0
        self.peak_score = 0.0

        if not os.path.isfile(self.model_path):
            message = f"找不到 openWakeWord 模型：{self.model_path}"
            self._log.warning("%s → 回退既有 KWS", message)
            raise OwwUnavailable(message)

        try:
            from openwakeword.model import Model

            kwargs = {
                "wakeword_models": [self.model_path],
                "inference_framework": "onnx",
            }
            if self.vad_threshold > 0:
                kwargs["vad_threshold"] = self.vad_threshold
            self._model = Model(**kwargs)
            self.model_names = list(self._model.models.keys())
            if not self.model_names:
                raise RuntimeError("模型未提供任何喚醒詞輸出")
        except Exception as exc:  # noqa: BLE001 - 需讓 daemon 能安全回退
            message = f"openWakeWord 載入失敗（{self.model_path}）：{exc}"
            self._log.warning("%s → 回退既有 KWS", message)
            raise OwwUnavailable(message) from exc

        self._log.info(
            "喚醒詞引擎：openWakeWord（%s，強命中門檻 %.2f；弱命中 %.2f x%d 幀 / %d 幀視窗；VAD %.2f）",
            self.model_names,
            self.threshold,
            self.relaxed_threshold,
            self.relaxed_hits,
            self.window_frames,
            self.vad_threshold,
        )

    @staticmethod
    def _to_pcm16(block: np.ndarray) -> np.ndarray:
        """正規化任意數值 block 成 openWakeWord 需要的 int16 PCM。"""
        samples = np.asarray(block).reshape(-1)
        if samples.dtype == np.int16:
            return samples
        safe = np.nan_to_num(
            samples.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return (np.clip(safe, -1.0, 1.0) * 32767.0).astype(np.int16)

    @staticmethod
    def _remove_dc(frame: np.ndarray) -> np.ndarray:
        """移除 DC offset（直流偏移）後再送模型。

        2026-09-28 根因：這支 Generalplus USB 麥克風的硬體增益全開（+33dB）＋
        Auto Gain Control 開啟時，輸出帶著 **0.008~0.015 的固定直流偏移**，
        AC 成分（真正的聲音）只有 0.0003。openWakeWord 的 melspectrogram 前端
        會把這個偏移當成訊號，於是安靜的房間也能穩定跑出 0.85~0.97 的「強命中」
        —— 整晚 2~7 點無人講話卻喚醒 80 幾次就是這麼來的，**調門檻治不了**
        （真喊也是 0.85~0.97，兩者完全重疊）。
        移掉 DC 之後，安靜時分數才會回到 0.0x。
        """
        samples = frame.astype(np.float32, copy=False)
        centered = samples - float(samples.mean())
        return np.clip(centered, -32768.0, 32767.0).astype(np.int16)

    def _score(self, frame: np.ndarray) -> float:
        """回傳這一幀所有輸出中最高的模型分數。"""
        predictions = self._model.predict(frame)
        return max((float(score) for score in predictions.values()), default=0.0)

    def relaxed_count(self) -> int:
        """目前視窗內有幾幀 >= relaxed_threshold（觀測用）。"""
        return sum(1 for s in self._recent if s >= self.relaxed_threshold)

    def feed(self, block: np.ndarray) -> bool:
        """餵入 16 kHz / mono / int16 音訊 block；確認命中時回傳 ``True``。"""
        pcm = self._to_pcm16(block)
        if pcm.size == 0:
            return False
        self._buffer = (
            pcm.copy()
            if self._buffer.size == 0
            else np.concatenate((self._buffer, pcm))
        )

        while self._buffer.size >= FRAME_SAMPLES:
            frame = self._buffer[:FRAME_SAMPLES]
            self._buffer = self._buffer[FRAME_SAMPLES:]
            frame = self._remove_dc(frame)
            if self.min_ac_rms > 0.0:
                ac_rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32))))) / 32768.0
                if ac_rms < self.min_ac_rms:
                    # 死訊號／純 DC：不送模型，分數直接記 0。
                    self.latest_score = 0.0
                    self.consecutive_frames = 0
                    self._recent.append(0.0)
                    continue
            score = self._score(frame)
            self.latest_score = score
            self.peak_score = max(self.peak_score, score)
            if score >= self.threshold:
                self.consecutive_frames += 1
            else:
                self.consecutive_frames = 0
            self._recent.append(score)

            strong = score >= self.threshold
            relaxed = self.relaxed_count() >= self.relaxed_hits
            if strong or relaxed:
                score_at_hit = self.latest_score
                self.reset()
                self.latest_score = score_at_hit
                return True
        return False

    def reset(self) -> None:
        """清空音訊/分數狀態，回到可偵測下一次喚醒的狀態。"""
        self._buffer = np.zeros(0, dtype=np.int16)
        self._recent.clear()
        self.consecutive_frames = 0
        self.latest_score = 0.0
        self.peak_score = 0.0
        self._model.reset()
