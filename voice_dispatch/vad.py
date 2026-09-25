"""能量 VAD（Voice Activity Detection）錄音狀態機。

把偵測邏輯抽成純狀態機 VadSegmenter，方便單元測試（不需真的麥克風）：
對每個區塊算 RMS，餵入 feed(rms, t)，回傳目前狀態。

狀態語意：
- "listening"：還沒偵測到語音，前置靜音等待中
- "speaking" ：偵測到語音，錄音中
- "done"     ：講完了（尾端靜音超時，或達到最長上限）
- "timeout"  ：前置靜音等太久都沒人講話，放棄
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from .config import VadConfig


class VadState(str, Enum):
    LISTENING = "listening"
    SPEAKING = "speaking"
    DONE = "done"
    TIMEOUT = "timeout"


def block_rms(block: np.ndarray) -> float:
    arr = np.asarray(block, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


class VadSegmenter:
    """語音起訖判定狀態機。"""

    def __init__(self, cfg: VadConfig):
        self.cfg = cfg
        self._first_t: Optional[float] = None       # 第一個區塊時間（算前置靜音）
        self._speech_start_t: Optional[float] = None
        self._last_voice_t: Optional[float] = None
        self._started = False
        self._finished = False

    @property
    def started(self) -> bool:
        return self._started

    def feed(self, rms: float, t: float, voiced: Optional[bool] = None) -> VadState:
        """餵入一個區塊的 RMS 與時間戳（秒），回傳目前狀態。

        voiced 有明確值時直接採用（例如改用 Silero 神經網路 VAD 的判定），
        否則用 `speech_rms_threshold` 這個 RMS 門檻判斷。
        一旦回傳 DONE / TIMEOUT，之後再呼叫會維持該終態。
        """
        cfg = self.cfg
        if self._finished:
            # 已結束，維持終態（起碼要有一次語音才算 DONE）
            return VadState.DONE if self._started else VadState.TIMEOUT

        if self._first_t is None:
            self._first_t = t

        if voiced is None:
            voiced = rms >= cfg.speech_rms_threshold

        if not self._started:
            if voiced:
                self._started = True
                self._speech_start_t = t
                self._last_voice_t = t
                return VadState.SPEAKING
            # 前置靜音等待
            if (t - self._first_t) >= cfg.preroll_timeout_sec:
                self._finished = True
                return VadState.TIMEOUT
            return VadState.LISTENING

        # 已在錄音中
        if voiced:
            self._last_voice_t = t

        # 注意：起始時間可能正好是 0.0，不能用「x or t」判斷（0.0 為 falsy）
        start = self._speech_start_t if self._speech_start_t is not None else t
        last_voice = self._last_voice_t if self._last_voice_t is not None else t

        duration = t - start
        if duration >= cfg.max_record_sec:
            self._finished = True
            return VadState.DONE

        silence = t - last_voice
        if silence >= cfg.trailing_silence_sec and duration >= cfg.min_record_sec:
            self._finished = True
            return VadState.DONE

        return VadState.SPEAKING
