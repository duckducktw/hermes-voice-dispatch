"""kws.py：喚醒詞偵測與 Silero VAD 的基本行為（不需麥克風）。"""
import os

import numpy as np
import pytest

from voice_dispatch import kws

VOSK = os.path.expanduser(
    "~/.local/share/hermes-voice-dispatch/vosk-model-small-en-us-0.15"
)


def test_silence_does_not_trigger_wake():
    """純靜音不該命中——常開聽最重要的性質就是不要亂觸發。"""
    sp = kws.WakeWordSpotter(["hey_jarvis"], threshold=0.5)
    for _ in range(20):        # 20 × 80ms = 1.6 秒
        assert sp.feed(np.zeros(1280, dtype=np.float32)) is None


def test_chunker_only_emits_full_chunks():
    """碎片不推論：湊滿一個 80ms 區塊才送模型。"""
    c = kws._Chunker()
    assert c.push(np.zeros(700, dtype=np.float32)) == []
    assert len(c.push(np.zeros(700, dtype=np.float32))) == 1


def test_silero_none_until_full_chunk():
    """Silero 只吃剛好 1280 samples；不足時回 None（上層退回 RMS 門檻）。"""
    v = kws.SileroVad(0.5)
    assert v.feed(np.zeros(512, dtype=np.float32)) is None
    assert v.feed(np.zeros(1024, dtype=np.float32)) in (True, False)


@pytest.mark.skipif(not os.path.isdir(VOSK), reason="需要 vosk 模型（未下載）")
def test_vosk_silence_no_hit():
    """Vosk 常開餵靜音不該命中喚醒詞。"""
    sp = kws.VoskSpotter(VOSK, words=["hermes"])
    for _ in range(20):
        assert sp.feed(np.zeros(1024, dtype=np.float32)) is None


# ── 詞序比對（喚醒詞＝"hey hermes"，2026-09-26）────────────────────────────
_HH = [("hey", "hermes")]        # 使用者定案的喚醒詞（token 序列）


def test_phrase_match_requires_full_phrase():
    """只喊 'hermes' 不算命中——要講全 'hey hermes'。"""
    assert kws.match_wake_phrase(["hermes"], [1.0], _HH, 0.9) is None
    assert kws.match_wake_phrase(["hey", "hermes"], [1.0, 1.0], _HH, 0.9) == "hey hermes"


def test_phrase_match_tolerates_leading_trailing_words():
    """前後有別的字（[unk]／雜訊）不影響，只看有沒有連著出現。"""
    assert kws.match_wake_phrase(
        ["[unk]", "hey", "hermes", "[unk]"], [0.5, 1.0, 1.0, 0.5], _HH, 0.9
    ) == "hey hermes"


def test_phrase_match_respects_confidence():
    """片段信心度低於門檻 → 不命中。"""
    assert kws.match_wake_phrase(["hey", "hermes"], [0.5, 1.0], _HH, 0.9) is None
    assert kws.match_wake_phrase(["hey", "hermes"], [1.0, 0.8], _HH, 0.9) is None


def test_phrase_match_reversed_order_no_hit():
    """詞序顛倒（hermes hey）不算命中。"""
    assert kws.match_wake_phrase(["hermes", "hey"], [1.0, 1.0], _HH, 0.9) is None


# ── 第二階段接受規則（WakeVerifier 用）：(前綴)(hermes變體) bigram ────────────
_PRE = ["hey", "hay", "a", "the", "he", "ok", "okay", "hi"]
_VAR = ["hermes", "homes", "hums", "hermis"]


def test_variants_accepts_true_wake():
    assert kws.match_wake_variants(["hey", "hermes"], [1.0, 1.0], _PRE, _VAR, 0.5) == "hey hermes"
    # "Hermes" 常被聽成 homes；"hey" 常被聽成 a/the — 都要接受
    assert kws.match_wake_variants(["hey", "homes"], [1.0, 1.0], _PRE, _VAR, 0.5) == "hey homes"
    assert kws.match_wake_variants(["a", "hermes"], [1.0, 1.0], _PRE, _VAR, 0.5) == "a hermes"


def test_variants_rejects_confusables():
    """近似音（hermit/her mess/'her mouse'）要被否決——否決力來自變體集合。"""
    assert kws.match_wake_variants(["hey", "hermit"], [1.0, 1.0], _PRE, _VAR, 0.5) is None
    assert kws.match_wake_variants(["hey", "her", "miss"], [1.0, 1.0, 1.0], _PRE, _VAR, 0.5) is None
    assert kws.match_wake_variants(["hey", "harm", "us"], [1.0, 1.0, 1.0], _PRE, _VAR, 0.5) is None
    # 只有 hermes 沒有前綴（單喊 hermes）不算命中
    assert kws.match_wake_variants(["hermes"], [1.0], _PRE, _VAR, 0.5) is None


def test_variants_respects_confidence():
    """信心度門檻套在**變體**上（前綴是功能詞，不套門檻）。"""
    assert kws.match_wake_variants(["hey", "hermes"], [1.0, 0.4], _PRE, _VAR, 0.5) is None
    # 前綴信心度低不影響（實測 "a hermes" 的 a 只有 0.44，仍要接受）
    assert kws.match_wake_variants(["a", "hermes"], [0.44, 1.0], _PRE, _VAR, 0.5) == "a hermes"
