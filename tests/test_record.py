"""_record_utterance 的 pre-roll 行為測試（避免 VAD 判定太晚而切掉字頭）。"""
import numpy as np
import pytest

from voice_dispatch import audio
from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher

BS = 1024
QUIET = np.zeros(BS, dtype=np.float32)
LOUD = np.full(BS, 0.3, dtype=np.float32)


def _install(monkeypatch, blocks):
    monkeypatch.setattr(audio, "read_blocks", lambda stream, bs: iter(blocks))


@pytest.fixture()
def disp():
    return VoiceDispatcher(Config(), dry_run=True)


def test_preroll_included(monkeypatch, disp):
    """語音開始前的安靜區塊要一起進來（不然字頭被切掉）。"""
    silence, speech, tail = 4, 20, 40
    _install(monkeypatch, [QUIET] * silence + [LOUD] * speech + [QUIET] * tail)
    out = disp._record_utterance(stream=None)
    assert out is not None
    assert out.size >= (silence + speech) * BS


def test_timeout_returns_none(monkeypatch, disp):
    """一直沒人講話（超過 preroll_timeout_sec）→ None。"""
    n = int(disp.cfg.vad.preroll_timeout_sec * 16000 / BS) + 5
    _install(monkeypatch, [QUIET] * n)
    assert disp._record_utterance(stream=None) is None


def test_silence_only_returns_none(monkeypatch, disp):
    """完全沒有語音 → 不該回傳一堆純靜音樣本。"""
    _install(monkeypatch, [QUIET] * 200)
    assert disp._record_utterance(stream=None) is None
