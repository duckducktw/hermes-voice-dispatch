"""cascade.WakeCascade：串接邏輯的單元測試（用 stub 取代真模型，不需要音訊/模型）。"""
import numpy as np
import pytest

from voice_dispatch import cascade as casc
from voice_dispatch.config import Config


class _Rec:
    def Reset(self):
        pass


class StubGate:
    """gate：第 `fire_at` 次 feed 命中一次，之後不再命中（預設讓 preroll 環先填滿）。"""
    def __init__(self, *a, **k):
        self.latest = "hey"
        self._rec = _Rec()
        self.n = 0
        self.fire_at = 8        # 8×64ms ≈ 0.5s，剛好填滿 preroll_sec=0.5 的環

    def reset_recognizer(self):
        pass

    def feed(self, block):
        self.n += 1
        return "hey" if self.n == self.fire_at else None


class StubVerifier:
    """verifier：前 confirm_after 次回 False，之後回 True。"""
    def __init__(self, *a, confirm_after=2, **k):
        self.latest = "hey hermes"
        self._rec = _Rec()
        self.confirm_after = confirm_after
        self.calls = 0

    def reset_recognizer(self):
        pass

    def verify(self, samples):
        self.calls += 1
        return self.calls >= self.confirm_after


def _make(monkeypatch, confirm_after):
    def _v(*a, **k):
        return StubVerifier(confirm_after=confirm_after)
    monkeypatch.setattr(casc.kws, "VoskSpotter", StubGate, raising=True)
    monkeypatch.setattr(casc.kws, "WakeVerifier", _v, raising=True)
    cfg = Config()
    cfg.audio.blocksize = 1024
    cfg.wake.preroll_sec = 0.5
    cfg.wake.verify_window_sec = 1.0
    cfg.wake.verify_interval_sec = 0.25
    return casc.WakeCascade(cfg)


def _run(det, n_blocks=200):
    for _ in range(n_blocks):
        r = det.feed(np.zeros(1024, dtype=np.float32))
        if r:
            return r
    return None


def test_cascade_gate_then_confirm_wakes(monkeypatch):
    """閘門命中後，確認器一旦通過就喚醒。"""
    det = _make(monkeypatch, confirm_after=2)
    assert _run(det) is not None


def test_cascade_never_confirmed_no_wake(monkeypatch):
    """閘門命中但確認器始終否決 → 不喚醒（退回監聽）。"""
    det = _make(monkeypatch, confirm_after=999)
    assert _run(det) is None


def test_cascade_confirm_waits_for_enough_audio(monkeypatch):
    """確認器第一次就被呼叫時，窗內音訊必須已含 preroll（不是空的）。"""
    seen = {}

    class Spy(StubVerifier):
        def verify(self, samples):
            seen.setdefault("first_len", len(samples))
            return super().verify(samples)

    monkeypatch.setattr(casc.kws, "VoskSpotter", StubGate, raising=True)
    monkeypatch.setattr(casc.kws, "WakeVerifier",
                        lambda *a, **k: Spy(confirm_after=1), raising=True)
    cfg = Config()
    cfg.audio.blocksize = 1024
    cfg.wake.preroll_sec = 0.5
    cfg.wake.verify_window_sec = 1.0
    cfg.wake.verify_interval_sec = 0.25
    det = casc.WakeCascade(cfg)
    _run(det)
    assert seen["first_len"] >= int(0.5 * 16000)   # 至少含 preroll
