"""openWakeWord 的 Hermes 喚醒詞包裝。

daemon 的音訊串流是 16 kHz / mono / int16，每次 1024 samples；openWakeWord
則以 80 ms（1280 samples）為最佳推論單位。本模組負責重新切塊、連續幀確認與
重置，讓呼叫端只要一直餵 daemon 的原始 block 即可。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz


class OwwUnavailable(RuntimeError):
    """openWakeWord 模型或執行期不可用。"""


class OwwSpotter:
    """以自訂 ONNX 模型偵測喚醒詞。

    ``feed()`` 回傳 ``True`` 前，必須收到連續 ``confirmation_frames`` 個分數
    大於等於 ``threshold`` 的 80 ms 音訊幀。命中後立即 reset，避免同一句話被
    重複喚醒。
    """

    def __init__(
        self,
        model_path: str,
        threshold: float,
        confirmation_frames: int,
        vad_threshold: float = 0.0,
        logger=None,
    ):
        self._log = logger or logging.getLogger(__name__)
        self.model_path = os.path.expanduser(model_path)
        self.threshold = float(threshold)
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.vad_threshold = float(vad_threshold)
        self._buffer = np.zeros(0, dtype=np.int16)
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
            "喚醒詞引擎：openWakeWord（%s，門檻 %.2f，連續 %d 幀，VAD %.2f）",
            self.model_names,
            self.threshold,
            self.confirmation_frames,
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

    def _score(self, frame: np.ndarray) -> float:
        """回傳這一幀所有輸出中最高的模型分數。"""
        predictions = self._model.predict(frame)
        return max((float(score) for score in predictions.values()), default=0.0)

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
            score = self._score(frame)
            self.latest_score = score
            self.peak_score = max(self.peak_score, score)
            if score >= self.threshold:
                self.consecutive_frames += 1
            else:
                self.consecutive_frames = 0
            if self.consecutive_frames >= self.confirmation_frames:
                score_at_hit = self.latest_score
                self.reset()
                self.latest_score = score_at_hit
                return True
        return False

    def reset(self) -> None:
        """清空音訊/分數狀態，回到可偵測下一次喚醒的狀態。"""
        self._buffer = np.zeros(0, dtype=np.int16)
        self.consecutive_frames = 0
        self.latest_score = 0.0
        self.peak_score = 0.0
        self._model.reset()
