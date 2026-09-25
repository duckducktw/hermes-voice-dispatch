"""雙拍手偵測（純函式 / 純狀態機，吃 numpy 區塊，可單元測試）。

原理：拍手是「短促、尖銳的能量瞬變」。我們對每個音訊區塊算 RMS，
以近期 RMS 中位數 × 倍數當自適應門檻；一個「上升緣」（前一塊在門檻下、
這一塊躍上門檻）視為一次瞬變。兩次瞬變間隔落在 [min_gap, max_gap] → 雙拍手。

只做 RMS（numpy）運算，CPU 閒置成本極低，不會持續跑 STT。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .config import ClapConfig


def block_rms(block: np.ndarray) -> float:
    """計算一個音訊區塊的 RMS（root mean square）。空區塊回傳 0。"""
    arr = np.asarray(block, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


class ClapDetector:
    """串流式雙拍手偵測。

    使用方式：對每個進來的區塊呼叫 process_block(block, timestamp)，
    回傳 True 表示「剛剛完成一次雙拍手」。timestamp 單位為秒。
    """

    def __init__(self, cfg: ClapConfig, samplerate: int, blocksize: int):
        self.cfg = cfg
        self.samplerate = samplerate
        self.blocksize = blocksize

        # 中位數視窗長度（以區塊數計）
        block_dur = blocksize / float(samplerate) if samplerate else 0.0
        n = int(round(cfg.rms_window_sec / block_dur)) if block_dur > 0 else 50
        self._history: deque = deque(maxlen=max(5, n))

        self._prev_loud = False               # 上一塊是否在門檻之上（判上升緣）
        self._last_transient_t: Optional[float] = None   # 最近一次瞬變時間
        self._first_clap_t: Optional[float] = None       # 第一次拍手時間（等待第二次）

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置雙拍手等待狀態（不清空 RMS 歷史）。"""
        self._first_clap_t = None

    def current_threshold(self) -> float:
        """目前的自適應門檻（給 README 校正與除錯用）。"""
        if self._history:
            median = float(np.median(self._history))
        else:
            median = 0.0
        return max(self.cfg.abs_floor, median * self.cfg.threshold_mult)

    # ------------------------------------------------------------------
    def process_block(self, block: np.ndarray, timestamp: float) -> bool:
        """處理一個區塊，回傳是否完成雙拍手。"""
        rms = block_rms(block)

        threshold = self.current_threshold()
        # 是否為「夠大聲但沒破音」的能量
        loud = (rms >= threshold) and (rms <= self.cfg.abs_ceil)

        # 只有非瞬變的區塊才餵進歷史，避免拍手把中位數拉高
        if not loud:
            self._history.append(rms)

        is_rising_edge = loud and not self._prev_loud
        self._prev_loud = loud

        detected = False
        if is_rising_edge:
            detected = self._register_transient(timestamp)

        return detected

    # ------------------------------------------------------------------
    def _register_transient(self, t: float) -> bool:
        """登記一次瞬變，判斷是否構成雙拍手。"""
        cfg = self.cfg

        # 不反應期：距離上一次瞬變太近，視為同一拍手的抖動
        if (
            self._last_transient_t is not None
            and (t - self._last_transient_t) < cfg.refractory_sec
        ):
            return False

        self._last_transient_t = t

        if self._first_clap_t is None:
            # 這是第一次拍手，開始等第二次
            self._first_clap_t = t
            return False

        gap = t - self._first_clap_t
        if gap < cfg.min_gap_sec:
            # 太快，可能還是同一拍手；忽略，維持等待
            return False
        if gap <= cfg.max_gap_sec:
            # 命中雙拍手
            self._first_clap_t = None
            return True
        # 間隔太長：把這次當成新的第一拍
        self._first_clap_t = t
        return False
