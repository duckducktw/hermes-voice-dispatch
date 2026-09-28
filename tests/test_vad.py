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


# ── early-giveup：語音佔比太低就提早收（2026-09-28）──────────────────────
# 使用者：「如果沒說話就把時間縮短」。症狀：媒體／環境聲讓 VAD 斷斷續續 voiced，
# 於是一路錄滿 max_record_sec(60s)（實測 60.10s）→ 乾等 + 多一張假任務卡。


def test_early_giveup_on_sparse_voice():
    """斷斷續續的環境聲（每 10 塊才 1 塊 voiced，佔比 ~10%）→ 12s 左右就收，不等到 60s。"""
    cfg = _cfg()
    seg = VadSegmenter(cfg)
    t = 0.0
    seg.feed(0.1, t)                 # 起頭要有一次語音才會進錄音狀態
    done_at = None
    for i in range(1, 1200):
        t += 0.064
        voiced = (i % 10 == 0)       # 稀疏語音
        if seg.feed(0.1 if voiced else 0.0, t, voiced=voiced) == VadState.DONE:
            done_at = t
            break
    assert done_at is not None
    # 必須遠早於 max_record_sec；trailing_silence 也可能先切，兩者都算「提早收」
    assert done_at < cfg.max_record_sec / 2


def test_early_giveup_does_not_kill_real_speech():
    """真的連續講話（佔比 100%）→ 不會被 early-giveup 誤殺，可一路講到 60s 上限。"""
    cfg = _cfg()
    seg = VadSegmenter(cfg)
    t = 0.0
    seg.feed(0.1, t)
    last = None
    while t < cfg.early_giveup_sec + 5.0:
        t += 0.064
        last = seg.feed(0.1, t, voiced=True)
        if last == VadState.DONE:
            break
    # 講到超過 early_giveup_sec 還是 SPEAKING（沒有被提早切掉）
    assert last == VadState.SPEAKING
