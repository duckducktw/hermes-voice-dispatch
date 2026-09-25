"""雙拍手偵測測試：合成尖峰訊號驗證命中/不命中/不誤觸。"""

import numpy as np

from voice_dispatch.clap import ClapDetector, block_rms
from voice_dispatch.config import Config
from voice_dispatch.wake import run_clap_loop

BS = 1024  # blocksize（與預設一致）


def _silence(n=1):
    return [np.zeros(BS, dtype=np.float32) for _ in range(n)]


def _clap():
    # 單一區塊的高能量瞬變（rms=0.5，落在 floor 與 ceil 之間）
    return [np.full(BS, 0.5, dtype=np.float32)]


def _run(blocks):
    cfg = Config()  # 預設 samplerate=16000, blocksize=1024 → 每塊 0.064s
    return run_clap_loop(iter(blocks), cfg)


def test_block_rms_basic():
    assert block_rms(np.zeros(10)) == 0.0
    assert abs(block_rms(np.full(10, 0.5)) - 0.5) < 1e-9
    assert block_rms(np.zeros(0)) == 0.0


def test_double_clap_hits():
    # 兩個瞬變間隔 5 塊 × 0.064 = 0.32s，落在 [0.12, 1.5] → 命中
    blocks = _silence(10) + _clap() + _silence(4) + _clap() + _silence(10)
    assert _run(blocks) is True


def test_single_clap_misses():
    blocks = _silence(10) + _clap() + _silence(10)
    assert _run(blocks) is False


def test_gap_too_long_misses():
    # 間隔 40 塊 × 0.064 = 2.56s > 1.5s → 不命中
    blocks = _silence(10) + _clap() + _silence(40) + _clap() + _silence(5)
    assert _run(blocks) is False


def test_noise_does_not_trigger():
    # 持續中低能量雜訊（rms≈0.02 < abs_floor 0.05）→ 從不算瞬變
    rng = np.random.default_rng(42)
    blocks = [rng.normal(0, 0.02, BS).astype(np.float32) for _ in range(80)]
    assert _run(blocks) is False


def test_sustained_loud_tone_not_double():
    # 一段持續大聲（非短促）：只有一個上升緣 → 不構成雙拍手
    blocks = _silence(5) + [np.full(BS, 0.5, dtype=np.float32) for _ in range(20)] + _silence(5)
    assert _run(blocks) is False


def test_detector_threshold_has_floor():
    det = ClapDetector(Config().clap, 16000, BS)
    # 尚無歷史時，門檻至少等於 abs_floor
    assert det.current_threshold() >= Config().clap.abs_floor
