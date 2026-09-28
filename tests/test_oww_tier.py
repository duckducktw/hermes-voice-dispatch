"""雙層喚醒判定測試（強命中 / 弱命中）——用腳本化分數餵 OwwSpotter，不打麥克風。

背景（2026-09-27 使用者三輪回饋）：
  1. 「太嚴格」→ 原本 0.7 / 連續 3 幀；log 顯示 0.938 的命中只維持 1 幀 → 被丟掉。
  2. 「還是太嚴格」→ 放成 0.45 / 單幀；0.9x 能醒，但單幀門檻太低。
  3. 「太鬆了，緊點」→ 定案**雙層**：
       強命中：單幀 >= threshold（真命中 0.9x → 秒醒）
       弱命中：window_frames 幀內 >= relaxed_hits 幀 >= relaxed_threshold
     單一雜訊尖峰（一幀 0.4~0.85、鄰居都低）不再觸發。
  4. 「看影片什麼都沒說就被回」→ 弱門檻 0.40 太寬（影片聲常態 0.40~0.69）→
     **弱門檻 0.40 → 0.60**（真喊 0.85~0.97、弱喊 >=0.75，切在斷層上）。
     → 本檔的 weak 測試值同步改成 0.65/0.70；並新增「0.4x 兩幀不再觸發」。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_dispatch.oww import OwwSpotter

MODEL = Path("~/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx").expanduser()
BLOCK = np.zeros(1280, dtype=np.int16)   # 剛好一幀（80ms @16k）


def _spotter_with_scores(monkeypatch, scores, **kw):
    if not MODEL.is_file():
        pytest.skip("需要本機 hey_hermes openWakeWord 模型")
    spotter = OwwSpotter(
        str(MODEL),
        kw.get("threshold", 0.85),
        0.0,
        relaxed_threshold=kw.get("relaxed", 0.60),
        window_frames=kw.get("window", 6),
        relaxed_hits=kw.get("hits", 2),
        # 這些測試用 monkeypatch 直接餵分數、音訊是合成零幀，
        # 會被 min_ac_rms 靜音閘擋掉 → 測分層判定時顯式停用它。
        min_ac_rms=kw.get("min_ac_rms", 0.0),
    )
    it = iter(scores)
    monkeypatch.setattr(spotter, "_score", lambda frame: next(it, 0.0))
    return spotter


def _feed(spotter, n):
    return [spotter.feed(BLOCK) for _ in range(n)]


def test_strong_single_frame_fires_immediately(monkeypatch):
    """單幀 0.9x（真命中的典型分數）→ 第一幀就該醒。"""
    spotter = _spotter_with_scores(monkeypatch, [0.94])
    assert _feed(spotter, 1) == [True]


def test_single_noise_spike_does_not_fire(monkeypatch):
    """單一雜訊尖峰：一幀 0.70、鄰居 0.05 → 不該醒（第三輪「太鬆」的主因）。"""
    scores = [0.05, 0.05, 0.70, 0.05, 0.05, 0.05, 0.05, 0.05]
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert not any(_feed(spotter, len(scores)))


def test_relaxed_two_frames_fire(monkeypatch):
    """弱命中：視窗內兩幀 0.65/0.70（前後都有證據）→ 第二幀醒。"""
    scores = [0.65, 0.70]
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert _feed(spotter, 2) == [False, True]


def test_video_level_two_frames_no_longer_fire(monkeypatch):
    """第四輪重點：影片聲常態 0.4x~0.5x，即使連兩幀也不再觸發（門檻 0.60）。"""
    scores = [0.43, 0.50, 0.45, 0.52, 0.41, 0.48, 0.05, 0.05]
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert not any(_feed(spotter, len(scores)))


def test_relaxed_frames_outside_window_do_not_fire(monkeypatch):
    """兩次弱命中被拉開超過視窗（6 幀）→ 不該醒。"""
    scores = [0.65] + [0.0] * 6 + [0.65] + [0.0] * 3
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert not any(_feed(spotter, len(scores)))


def test_below_relaxed_never_fires(monkeypatch):
    """連續低分（環境底 0.05~0.34）→ 永不觸發。"""
    scores = [0.34] * 20
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert not any(_feed(spotter, len(scores)))


# ── 靜音閘（min_ac_rms）回歸，2026-09-28 ────────────────────────────────
# 根因：麥克風變死訊號（常數 DC、AC≈0）時 openWakeWord 仍穩定吐 0.85~0.97 的
# 假強命中 → 9/27 晚~9/28 共 237 次喚醒只有 14 次有真內容。真喊的 AC-RMS 約
# 0.06，死訊號約 0.0003（差 200 倍），所以用「有沒有訊號」當閘門，不動門檻。


def _dc_frame(dc: float = 0.008) -> np.ndarray:
    """模擬死訊號：純直流、完全沒有 AC 成分。"""
    return np.full(1280, int(dc * 32768), dtype=np.int16)


def _speech_frame(amp: float = 0.06) -> np.ndarray:
    """模擬真實語音電平的 AC 訊號。"""
    t = np.arange(1280) / 16000.0
    return (np.sin(2 * np.pi * 300 * t) * amp * 32767).astype(np.int16)


def test_dc_only_frames_never_fire(monkeypatch):
    """死訊號（純 DC）即使模型吐 0.95，也因為沒有 AC 訊號而不觸發。"""
    spotter = _spotter_with_scores(monkeypatch, [0.95] * 10, min_ac_rms=0.0012)
    assert not any(spotter.feed(_dc_frame()) for _ in range(10))


def test_real_level_speech_still_fires(monkeypatch):
    """真實語音電平不會被靜音閘誤殺（避免矯枉過正把真喊擋掉）。"""
    spotter = _spotter_with_scores(monkeypatch, [0.95], min_ac_rms=0.0012)
    assert spotter.feed(_speech_frame()) is True
