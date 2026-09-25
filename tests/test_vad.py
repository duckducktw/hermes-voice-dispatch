"""能量 VAD 狀態機測試：靜音切斷、最長上限、前置靜音逾時。"""

from voice_dispatch.config import Config
from voice_dispatch.vad import VadSegmenter, VadState


def _cfg():
    return Config().vad


def test_trailing_silence_cuts():
    seg = VadSegmenter(_cfg())
    # 先講話 0.5s
    t = 0.0
    state = seg.feed(0.1, t)
    assert state == VadState.SPEAKING
    for _ in range(8):
        t += 0.064
        seg.feed(0.1, t)
    # 開始靜音，持續超過 trailing_silence_sec(1.2s) → DONE
    last_state = None
    for _ in range(40):
        t += 0.064
        last_state = seg.feed(0.0, t)
        if last_state == VadState.DONE:
            break
    assert last_state == VadState.DONE
    assert seg.started is True


def test_max_record_caps():
    cfg = _cfg()
    seg = VadSegmenter(cfg)
    t = 0.0
    seg.feed(0.1, t)  # 起始語音
    state = None
    # 持續語音直到超過 max_record_sec
    while t < cfg.max_record_sec + 1.0:
        t += 0.1
        state = seg.feed(0.1, t)
        if state == VadState.DONE:
            break
    assert state == VadState.DONE
    assert t >= cfg.max_record_sec


def test_preroll_timeout():
    cfg = _cfg()
    seg = VadSegmenter(cfg)
    t = 0.0
    state = None
    while t < cfg.preroll_timeout_sec + 1.0:
        state = seg.feed(0.0, t)  # 全程靜音
        if state == VadState.TIMEOUT:
            break
        t += 0.1
    assert state == VadState.TIMEOUT
    assert seg.started is False


def test_terminal_state_is_sticky():
    seg = VadSegmenter(_cfg())
    seg.feed(0.0, 0.0)
    # 一路靜音到逾時
    t = 0.0
    while seg.feed(0.0, t) != VadState.TIMEOUT:
        t += 0.5
        if t > 20:
            break
    # 之後再餵仍維持 TIMEOUT（never started）
    assert seg.feed(0.5, t + 1) == VadState.TIMEOUT
