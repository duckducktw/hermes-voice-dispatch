"""_record_utterance 的行為測試：pre-roll 字頭回補 + settle drain。

注意：`_record_utterance` 先用 `stream.read()` 做 settle drain（排掉播放提示語
期間積在緩衝的自家人聲），之後才讀 VAD 區塊，所以假 stream 必須實作 `read()`。
"""
import numpy as np
import pytest

from voice_dispatch import audio
from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher

BS = 1024
QUIET = np.zeros(BS, dtype=np.float32)
LOUD = np.full(BS, 0.3, dtype=np.float32)


class _FakeStream:
    """只實作 _drain() 會用到的 read()。"""

    def read(self, frames):
        return np.zeros((frames, 1), dtype=np.float32), False


@pytest.fixture()
def disp():
    return VoiceDispatcher(Config(), dry_run=True)


def _run(monkeypatch, disp, blocks):
    monkeypatch.setattr(audio, "read_blocks", lambda stream, bs: iter(blocks))
    return disp._record_utterance(_FakeStream())


def test_preroll_included(monkeypatch, disp):
    """語音前的安靜區塊要一起收進來（避免 VAD 判定太晚切掉字頭）。"""
    blocks = [QUIET] * 4 + [LOUD] * 20 + [QUIET] * 30
    out = _run(monkeypatch, disp, blocks)
    assert out is not None
    assert out.size >= (4 + 20) * BS


def test_silence_only_returns_none(monkeypatch, disp):
    """整段都安靜 → 沒有需求，回 None。"""
    out = _run(monkeypatch, disp, [QUIET] * 400)
    assert out is None


def test_timeout_returns_none(monkeypatch, disp):
    """前置靜音等太久 → 回 None（呼叫端會重問）。"""
    disp.cfg.vad.preroll_timeout_sec = 0.1
    out = _run(monkeypatch, disp, [QUIET] * 400)
    assert out is None
