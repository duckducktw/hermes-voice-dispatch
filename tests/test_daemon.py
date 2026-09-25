"""Daemon 的「只聽到否定詞才重錄」判定（fail-open 確認輪）測試。

2026-09-25：確認輪不再要求使用者說「對」，只有整句就是否定／重來詞才重錄。
"""
import pytest

from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher


@pytest.fixture()
def d():
    return VoiceDispatcher(Config(), dry_run=True)


@pytest.mark.parametrize("text", ["不對", "重來", "再說", "錯", "不要", "不對不對"])
def test_retry_only(d, text):
    assert d._is_retry_only(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "幫我把伺服器重啟",
        "我覺得這不行，換一個做法",   # 含「不」但不是單純否定 → 不可誤判成重錄
        "不要重啟，改成先備份",       # 有實際需求內容 → 不可誤判
        "",
    ],
)
def test_not_retry_only(d, text):
    assert d._is_retry_only(text) is False


# --------------------------------------------------------------------------
# 提示聲（2026-09-25 使用者指定）：
#   接收到喚醒 → 一個咚咚；聽完需求（有講或沒講都算）→ 再一個咚咚。
#   整場不講話（chime 模式是預設）。
# --------------------------------------------------------------------------
def _spy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "voice_dispatch.daemon.tts.play_chime",
        lambda cfg, logger=None, cue="start": calls.append(("chime", cue)) or True,
    )
    monkeypatch.setattr(
        "voice_dispatch.daemon.tts.speak", lambda *a, **k: calls.append(("speak", None)) or True
    )
    monkeypatch.setattr(
        "voice_dispatch.daemon.tts.play_beep", lambda *a, **k: calls.append(("beep", None)) or True
    )
    return calls


def test_chime_mode_plays_chime_not_speech(monkeypatch):
    """預設 chime 模式：開始／結束各一聲，完全不講話；且兩者帶不同 cue。"""
    calls = _spy(monkeypatch)
    d = VoiceDispatcher(Config(), dry_run=True)
    d._cue_start(0)
    d._cue_end()
    assert calls == [("chime", "start"), ("chime", "end")]


def test_prompt_and_capture_chimes_even_when_nothing_said(monkeypatch):
    """沒講任何內容而結束（VAD 逾時）→ 一樣要再一聲（end）。"""
    calls = _spy(monkeypatch)
    d = VoiceDispatcher(Config(), dry_run=True)
    monkeypatch.setattr(d, "_record_utterance", lambda stream: None)
    assert d.prompt_and_capture(object(), 0) is None
    assert calls == [("chime", "start"), ("chime", "end")]


def test_voice_mode_still_speaks(monkeypatch):
    """prompt_mode="voice" 要保留舊行為（TTS 引導語），不能被 chime 蓋掉。"""
    calls = _spy(monkeypatch)
    cfg = Config()
    cfg.tts.prompt_mode = "voice"
    d = VoiceDispatcher(cfg, dry_run=True)
    d._cue_start(0)
    d._cue_end()
    assert calls == [("speak", None)]
