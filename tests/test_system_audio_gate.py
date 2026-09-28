"""system_audio_playing / _echo_muted 第③條閘門的單元測試（2026-09-28）。

使用者：「要排除電腦發出的聲音」。YouTube/Discord/Minecraft 從喇叭出去的人聲
被麥克風收回來會誤喚醒，①② 的 echo_guard 只管 daemon 自己播的東西。
"""

from voice_dispatch import audio
from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher


def _d(**wake):
    cfg = Config()
    for k, v in wake.items():
        setattr(cfg.wake, k, v)
    return VoiceDispatcher(cfg, dry_run=True)


def test_system_audio_gate_blocks_when_playing(monkeypatch):
    """電腦在出聲 → 喚醒閘關上。"""
    d = _d(mute_while_system_audio=True, single_flight=False, echo_guard_sec=0.0)
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "system_audio_playing", lambda **kw: True)
    monkeypatch.setattr(audio, "system_audio_recent_sec", lambda: 0.0)
    assert d._echo_muted() is True


def test_system_audio_gate_sticky_window(monkeypatch):
    """對白停頓：當下量到沒聲音，但 1.5s 內剛出聲過 → 仍要擋。"""
    d = _d(mute_while_system_audio=True, single_flight=False,
           echo_guard_sec=0.0, system_audio_guard_sec=1.5)
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "system_audio_playing", lambda **kw: False)
    monkeypatch.setattr(audio, "system_audio_recent_sec", lambda: 0.4)
    assert d._echo_muted() is True


def test_system_audio_gate_opens_when_quiet(monkeypatch):
    """電腦安靜夠久 → 閘門放行（不能永遠關著，否則整天叫不醒）。"""
    d = _d(mute_while_system_audio=True, single_flight=False,
           echo_guard_sec=0.0, system_audio_guard_sec=1.5)
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "system_audio_playing", lambda **kw: False)
    monkeypatch.setattr(audio, "system_audio_recent_sec", lambda: 99.0)
    assert d._echo_muted() is False


def test_system_audio_gate_can_be_disabled(monkeypatch):
    """關掉開關就完全不看系統音訊（回到舊行為）。"""
    d = _d(mute_while_system_audio=False, single_flight=False, echo_guard_sec=0.0)
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "system_audio_playing",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("不該被呼叫")))
    assert d._echo_muted() is False


def test_system_audio_playing_fails_open(monkeypatch):
    """偵測失敗（沒有 pactl/ffmpeg）→ 回 False（fail-open，寧可放行也不要叫不醒）。"""
    monkeypatch.setattr(audio, "_default_sink_monitor", lambda: "")
    assert audio.system_audio_playing(ttl=0) is False
