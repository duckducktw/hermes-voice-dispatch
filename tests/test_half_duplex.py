"""半雙工（自己在出聲時不做喚醒偵測）的回歸測試。

2026-09-26 使用者回報「說完需求後，過一下子會連響好幾聲咚咚」。
根因：語音回報的 TTS 是在**背景 thread** 播的，主迴圈同時間已經回到
`wait_for_wake()` 在聽 → 把自己的聲音（提示音／TTS）收進喚醒偵測 →
誤喚醒 → 又播一輪提示音 → 連響。

這裡釘住三件事：
  1. `audio` 的播放狀態（busy／quiet）正確。
  2. `daemon._echo_muted` 的判定（播放中 / guard 內 / guard 後 / 關閉）。
  3. 喚醒迴圈真的**不把出聲期間的 block 餵給偵測器**，且恢復時會 reset。
"""
import pytest

from voice_dispatch import audio
from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher


@pytest.fixture()
def d():
    return VoiceDispatcher(Config(), dry_run=True)


# --------------------------------------------------------------------------
# audio 的播放狀態
# --------------------------------------------------------------------------
def test_output_state_idle(monkeypatch):
    monkeypatch.setattr(audio, "_OUTPUT_ACTIVE", 0)
    monkeypatch.setattr(audio, "_OUTPUT_ENDED_AT", 0.0)
    assert audio.output_busy() is False
    assert audio.output_quiet_sec() == float("inf")


def test_play_file_marks_busy_then_quiet(monkeypatch):
    monkeypatch.setattr(audio, "_OUTPUT_ACTIVE", 0)
    monkeypatch.setattr(audio, "_OUTPUT_ENDED_AT", 0.0)
    seen = {}

    def fake_inner(path, cfg, logger=None):
        seen["busy_during"] = audio.output_busy()
        return True

    monkeypatch.setattr(audio, "_play_file_inner", fake_inner)
    assert audio.play_file("x.wav", Config().audio) is True
    assert seen["busy_during"] is True        # 播放期間＝忙碌
    assert audio.output_busy() is False       # 播完＝不忙
    assert audio.output_quiet_sec() < 1.0     # 但「剛播完」


def test_play_file_clears_busy_on_exception(monkeypatch):
    monkeypatch.setattr(audio, "_OUTPUT_ACTIVE", 0)
    monkeypatch.setattr(audio, "_OUTPUT_ENDED_AT", 0.0)

    def boom(path, cfg, logger=None):
        raise RuntimeError("player exploded")

    monkeypatch.setattr(audio, "_play_file_inner", boom)
    with pytest.raises(RuntimeError):
        audio.play_file("x.wav", Config().audio)
    assert audio.output_busy() is False       # 例外也不能卡在忙碌


# --------------------------------------------------------------------------
# daemon._echo_muted
# --------------------------------------------------------------------------
def test_echo_muted_true_while_playing(d, monkeypatch):
    monkeypatch.setattr(audio, "output_busy", lambda: True)
    monkeypatch.setattr(audio, "output_quiet_sec", lambda: 99.0)
    assert d._echo_muted() is True


def test_echo_muted_true_within_guard(d, monkeypatch):
    d.cfg.wake.echo_guard_sec = 0.8
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "output_quiet_sec", lambda: 0.2)
    assert d._echo_muted() is True


def test_echo_muted_false_after_guard(d, monkeypatch):
    d.cfg.wake.echo_guard_sec = 0.8
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "output_quiet_sec", lambda: 5.0)
    assert d._echo_muted() is False


def test_echo_muted_never_before_first_play(d, monkeypatch):
    d.cfg.wake.echo_guard_sec = 0.8
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "output_quiet_sec", lambda: float("inf"))
    assert d._echo_muted() is False


def test_echo_muted_disabled_when_guard_zero(d, monkeypatch):
    d.cfg.wake.echo_guard_sec = 0.0
    monkeypatch.setattr(audio, "output_busy", lambda: False)
    monkeypatch.setattr(audio, "output_quiet_sec", lambda: 0.0)
    assert d._echo_muted() is False


# --------------------------------------------------------------------------
# _skip_echo_blocks（給 clap 迴圈用的過濾器）
# --------------------------------------------------------------------------
def test_skip_echo_blocks_drops_muted(d, monkeypatch):
    calls = {"n": 0}

    def fake_muted():
        calls["n"] += 1
        return calls["n"] <= 3          # 前 3 個 block 是自己在出聲

    monkeypatch.setattr(d, "_echo_muted", fake_muted)
    assert list(d._skip_echo_blocks([1, 2, 3, 4, 5])) == [4, 5]


def test_skip_echo_blocks_stops_on_stop_flag(d, monkeypatch):
    monkeypatch.setattr(d, "_echo_muted", lambda: False)
    gen = d._skip_echo_blocks([1, 2, 3])
    assert next(gen) == 1
    d._stop = True
    assert list(gen) == []


# --------------------------------------------------------------------------
# 喚醒迴圈真的要擋掉（整合：假 stream／假 cascade）
# --------------------------------------------------------------------------
class _FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_wake_loop_ignores_echo_blocks(d, monkeypatch):
    fed = []
    resets = []

    class _Cascade:
        def __init__(self, *a, **k):
            pass

        def feed(self, block):
            fed.append(block)
            return "hey hermes（確認轉錄 'hey hermes'）" if block == 5 else None

        def reset(self, hard=False):
            resets.append(hard)

    monkeypatch.setattr("voice_dispatch.daemon.cascade.WakeCascade", _Cascade)
    monkeypatch.setattr(audio, "input_stream", lambda cfg: _FakeStream())
    monkeypatch.setattr(audio, "read_blocks_watched", lambda *a, **k: iter([1, 2, 3, 4, 5]))
    monkeypatch.setattr(d, "_log_stream_health", lambda *a, **k: None)

    flags = iter([True, True, True, False, False, False])
    monkeypatch.setattr(d, "_echo_muted", lambda: next(flags, False))

    assert d._wait_for_wake_kws() is True
    assert fed == [4, 5], "出聲期間的 block 不可以餵給喚醒偵測器"
    assert False in resets, "恢復監聽時要 soft reset 掉殘留的 ring"
