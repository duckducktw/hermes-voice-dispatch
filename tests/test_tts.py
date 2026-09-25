"""tts.muted()：靜音測試模式——存在 mute 檔就完全不合成、不播放 TTS。"""
import os

import numpy as np

from voice_dispatch import tts
from voice_dispatch.config import Config


def test_muted_reflects_file(tmp_path):
    cfg = Config()
    cfg.tts.mute_file = str(tmp_path / "mute")
    assert tts.muted(cfg) is False
    with open(cfg.tts.mute_file, "w"):
        pass
    assert tts.muted(cfg) is True
    os.remove(cfg.tts.mute_file)
    assert tts.muted(cfg) is False


def test_speak_returns_false_when_muted(tmp_path):
    """靜音時 speak() 直接回 False（不合成、不播放）——測試時最在意的性質。"""
    cfg = Config()
    cfg.tts.mute_file = str(tmp_path / "mute")
    with open(cfg.tts.mute_file, "w"):
        pass
    assert tts.speak("測試", cfg) is False


def test_muted_false_when_no_path():
    cfg = Config()
    cfg.tts.mute_file = ""
    assert tts.muted(cfg) is False


# --------------------------------------------------------------------------
# 「咚咚」提示音（2026-09-25 使用者指定：互動不要講話，只要兩聲咚咚）
# --------------------------------------------------------------------------
def test_make_chime_shape():
    """兩聲 + 中間的縫，長度與峰值都照 config 走。"""
    cfg = Config()
    sr = cfg.audio.samplerate
    n = int(round(cfg.tts.chime_tone_dur_sec * sr))
    gap = int(round(cfg.tts.chime_gap_sec * sr))
    s = tts.make_chime(cfg)
    assert s.size == 2 * n + gap
    assert s.dtype == np.float32
    # 兩聲都要真的有聲音（只有第 1 聲是常見的手殘 bug）
    assert float(np.abs(s[:n]).max()) > 0.05
    assert float(np.abs(s[n + gap:n + gap + n]).max()) > 0.05
    # 縫要是靜音
    assert float(np.abs(s[n:n + gap]).max()) == 0.0
    # 音量不超過設定值（避免刺耳/爆音）
    assert float(np.abs(s).max()) <= cfg.tts.chime_volume + 1e-6


def test_play_chime_returns_false_when_muted(tmp_path):
    cfg = Config()
    cfg.tts.mute_file = str(tmp_path / "mute")
    with open(cfg.tts.mute_file, "w"):
        pass
    assert tts.play_chime(cfg) is False

