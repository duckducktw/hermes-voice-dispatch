"""神經網路喚醒詞（openWakeWord）與神經網路 VAD（Silero）——真實語音助手的做法。

為什麼要換掉舊做法
------------------
舊做法是「拍手兩下 → 錄 2.5 秒 → 對那 2.5 秒做 STT → 比對字串裡有沒有 'hermes'」。
三個結構性缺點（2026-09-25 使用者實測反饋「太智障」）：

1. 慢——命中後還要 2.5s 錄音 + 2.6s STT 才開始回應。
   Home Assistant 官方文件講得很直白：
   「The wake words have to be processed extremely fast: You can't have a voice
    assistant start listening 5 seconds after a wake word is spoken.」
2. 脆——STT 對「短音訊 + 前後靜音」極易吐幻覺（實測回吐過
   「感謝收看。」「那個更難,那個更難,那個更難。」）。
3. 還要先拍手——完全不像在跟人講話。

真實助手（Alexa / Google / Siri / Home Assistant）都不對常開路徑做 STT，
而是跑一顆**微小的關鍵詞模型（KWS）**，每 80ms 推論一次，命中才把音訊送去 STT。

本模組提供兩個元件：
  WakeWordSpotter —— openWakeWord 包裝，每 80ms 判一次喚醒詞（延遲 < 0.1s）
  SileroVad       —— Silero 神經網路 VAD，取代「RMS 門檻」判斷有沒有人講話

兩者都吃 1280 samples（80ms @16kHz）的單位；本模組內建重新切塊，
所以呼叫端可以直接餵 daemon 的 1024-sample 區塊。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

CHUNK = 1280          # 80ms @16kHz：openWakeWord 與 Silero VAD 共同的推論單位


class _Chunker:
    """把任意長度的 float32 區塊累積並重新切成固定大小的區塊。"""

    def __init__(self, size: int = CHUNK):
        self.size = size
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, samples) -> List[np.ndarray]:
        a = np.asarray(samples, dtype=np.float32).ravel()
        if a.size:
            self._buf = a if self._buf.size == 0 else np.concatenate([self._buf, a])
        out: List[np.ndarray] = []
        while self._buf.size >= self.size:
            out.append(self._buf[: self.size])
            self._buf = self._buf[self.size:]
        return out


def _to_pcm16(chunk: np.ndarray) -> np.ndarray:
    """float32 [-1,1] → int16。NaN/爆音先夾住，避免模型吃到垃圾。"""
    safe = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
    return (np.clip(safe, -1.0, 1.0) * 32767.0).astype(np.int16)


class WakeWordSpotter:
    """openWakeWord 喚醒詞偵測。不用拍手、不對喚醒詞做 STT。"""

    def __init__(self, models: List[str], threshold: float = 0.5, logger=None):
        from openwakeword.model import Model   # 延遲載入（吃 onnxruntime）

        self._model = Model(
            wakeword_models=list(models), inference_framework="onnx"
        )
        self.models = list(self._model.models.keys())
        self.threshold = float(threshold)
        self.latest: Dict[str, float] = {name: 0.0 for name in self.models}
        self.peak: Dict[str, float] = {name: 0.0 for name in self.models}
        self._chunker = _Chunker()
        if logger:
            logger.info(
                "喚醒詞引擎：openWakeWord %s（門檻 %.2f，每 %.0fms 判一次）",
                self.models, self.threshold, CHUNK / 16000 * 1000,
            )

    def feed(self, samples) -> Optional[str]:
        """餵入音訊區塊；命中喚醒詞回傳模型名，否則 None。"""
        hit: Optional[str] = None
        for chunk in self._chunker.push(samples):
            for name, score in self._model.predict(_to_pcm16(chunk)).items():
                s = float(score)
                self.latest[name] = s
                if s > self.peak[name]:
                    self.peak[name] = s
                if s >= self.threshold and hit is None:
                    hit = name
        return hit


class SileroVad:
    """Silero 神經網路 VAD：判斷「這 80ms 有沒有人在講話」。

    比 RMS 門檻穩健得多——RMS 門檻要嘛太敏感（冷氣／風扇被判成語音），
    要嘛太保守（隔著桌面講話被判成沒人講話，程式就一直重問需求）。
    """

    def __init__(self, threshold: float = 0.5, logger=None):
        from openwakeword.vad import VAD

        self._vad = VAD()
        self.threshold = float(threshold)
        self.last_prob = 0.0
        self._chunker = _Chunker()
        if logger:
            logger.info("端點偵測：Silero VAD（門檻 %.2f）", self.threshold)

    def feed(self, samples) -> Optional[bool]:
        """餵入音訊區塊；有完整 80ms 區塊時回傳該區塊是否為語音，否則 None。"""
        verdict: Optional[bool] = None
        for chunk in self._chunker.push(samples):
            try:
                self.last_prob = float(self._vad.predict(_to_pcm16(chunk)))
            except Exception:  # noqa: BLE001 —— VAD 壞掉不該讓整個守護程式掛掉
                continue
            verdict = self.last_prob >= self.threshold
        return verdict
