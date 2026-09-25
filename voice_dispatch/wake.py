"""喚醒組裝：拍手偵測 → 錄喚醒詞視窗 → STT → 比對喚醒詞。

拍手偵測的串流迴圈需要真實麥克風，放在 daemon 用；這裡提供：
- verify_wake_word：純判定（委派 text 模組），可單元測試。
- run_clap_loop：吃一個「區塊產生器」跑拍手偵測，回傳偵測到雙拍手為止。
  （產生器讓測試可餵合成訊號，不必真的開麥克風。）
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import numpy as np

from .clap import ClapDetector
from .config import Config
from .text import is_wake_transcript


def verify_wake_word(transcript: str, cfg: Config) -> bool:
    """轉錄文字是否命中喚醒詞。"""
    return is_wake_transcript(transcript, cfg.wake.keywords)


def run_clap_loop(
    blocks: Iterator[np.ndarray],
    cfg: Config,
    should_stop: Optional[Callable[[], bool]] = None,
    time_fn: Optional[Callable[[], float]] = None,
) -> bool:
    """對區塊串流跑雙拍手偵測。

    偵測到雙拍手回傳 True；串流結束（產生器耗盡）或 should_stop() 為真回傳 False。
    time_fn 未提供時，以「已處理區塊數 × 區塊時長」推算時間戳（穩定、可測）。
    """
    detector = ClapDetector(cfg.clap, cfg.audio.samplerate, cfg.audio.blocksize)
    block_dur = cfg.audio.blocksize / float(cfg.audio.samplerate)
    idx = 0
    for block in blocks:
        if should_stop is not None and should_stop():
            return False
        t = time_fn() if time_fn is not None else idx * block_dur
        idx += 1
        if detector.process_block(block, t):
            return True
    return False
