"""雙層喚醒判定測試（強命中 / 弱命中）——用腳本化分數餵 OwwSpotter，不打麥克風。

背景（2026-09-27 使用者三輪回饋）：
  1. 「太嚴格」→ 原本 0.7 / 連續 3 幀；log 顯示 0.938 的命中只維持 1 幀 → 被丟掉。
  2. 「還是太嚴格」→ 放成 0.45 / 單幀；0.9x 能醒，但單幀門檻太低。
  3. 「太鬆了，緊點」→ 定案**雙層**：
       強命中：單幀 >= threshold（真命中 0.9x → 秒醒）
       弱命中：window_frames 幀內 >= relaxed_hits 幀 >= relaxed_threshold
     單一雜訊尖峰（一幀 0.4~0.85、鄰居都低）不再觸發。
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
        relaxed_threshold=kw.get("relaxed", 0.40),
        window_frames=kw.get("window", 6),
        relaxed_hits=kw.get("hits", 2),
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
    """單一雜訊尖峰：一幀 0.60、鄰居 0.05 → 不該醒（第三輪「太鬆」的主因）。"""
    scores = [0.05, 0.05, 0.60, 0.05, 0.05, 0.05, 0.05, 0.05]
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert not any(_feed(spotter, len(scores)))


def test_relaxed_two_frames_fire(monkeypatch):
    """弱命中：視窗內兩幀 0.45（前後都有證據）→ 第二幀醒。"""
    scores = [0.45, 0.50]
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert _feed(spotter, 2) == [False, True]


def test_relaxed_frames_outside_window_do_not_fire(monkeypatch):
    """兩次弱命中被拉開超過視窗（6 幀）→ 不該醒。"""
    scores = [0.45] + [0.0] * 6 + [0.45] + [0.0] * 3
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert not any(_feed(spotter, len(scores)))


def test_below_relaxed_never_fires(monkeypatch):
    """連續低分（環境底 0.05~0.34）→ 永不觸發。"""
    scores = [0.34] * 20
    spotter = _spotter_with_scores(monkeypatch, scores)
    assert not any(_feed(spotter, len(scores)))
