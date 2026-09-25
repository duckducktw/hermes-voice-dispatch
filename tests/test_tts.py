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
    """兩聲（含尾韻）：長度、起音能量、音量上限、尾端淡出都照 config 走。"""
    cfg = Config()
    sr = cfg.audio.samplerate
    n = int(round(cfg.tts.chime_tone_dur_sec * sr))
    gap = int(round(cfg.tts.chime_gap_sec * sr))
    tail = int(round(cfg.tts.chime_tail_sec * sr))
    s = tts.make_chime(cfg)
    assert s.size == 2 * n + gap + tail
    assert s.dtype == np.float32
    # 兩聲都要真的有起音（只有第 1 聲是常見的手殘 bug）
    assert float(np.abs(s[:n]).max()) > 0.05
    assert float(np.abs(s[n + gap:n + gap + n]).max()) > 0.05
    # 音量不超過設定值（避免刺耳/爆音）
    assert float(np.abs(s).max()) <= cfg.tts.chime_volume + 1e-6
    # 尾端要收乾淨，不能「啪」一聲
    assert abs(float(s[-1])) < 1e-3


def test_make_chime_is_not_a_hollow_sine():
    """「有質感」的回歸測試：頻譜不能只有基頻。

    純正弦的話能量幾乎全集中在基頻那一格（＝聽起來空洞的電子嗶）。
    FM 調變／泛音列一定會產生基頻以外的邊帶，所以檢查「基頻以外的能量占比」。
    """
    cfg = Config()
    sr = cfg.audio.samplerate
    n = int(round(cfg.tts.chime_tone_dur_sec * sr))
    s = tts.make_chime(cfg).astype(np.float64)[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(s))
    df = sr / n
    i0 = int(round(cfg.tts.chime_tone1_hz / df))
    band = slice(max(0, i0 - 3), i0 + 4)
    total = float(spec.sum())
    inside = float(spec[band].sum())
    assert total > 0
    assert (total - inside) > 0.15 * total, "基頻以外幾乎沒有能量 → 聽起來會是空洞的電子音"


def test_play_chime_returns_false_when_muted(tmp_path):
    cfg = Config()
    cfg.tts.mute_file = str(tmp_path / "mute")
    with open(cfg.tts.mute_file, "w"):
        pass
    assert tts.play_chime(cfg) is False


def test_make_chime_uses_file_asset_when_configured(tmp_path):
    """素材模式：chime_files 指到音檔時就播它（長度＝素材長度，音量歸一到設定值）。"""
    from voice_dispatch import audio

    cfg = Config()
    sr = cfg.audio.samplerate
    tone = np.sin(2 * np.pi * 440.0 * np.arange(int(0.25 * sr)) / sr).astype(np.float32)
    asset = tmp_path / "hit.wav"
    audio.write_wav(str(asset), tone, sr)
    cfg.tts.chime_source = "files"
    cfg.tts.chime_files = [str(asset)]
    s = tts.make_chime(cfg)
    assert s.size == tone.size
    assert abs(float(np.abs(s).max()) - cfg.tts.chime_volume) < 1e-3
    # 兩個檔案 → 中間插 chime_gap_sec
    cfg.tts.chime_files = [str(asset), str(asset)]
    gap = int(round(cfg.tts.chime_gap_sec * sr))
    assert tts.make_chime(cfg).size == 2 * tone.size + gap


def test_make_chime_falls_back_to_synth_when_asset_missing(tmp_path):
    """素材不存在（例如別台機器）→ 自動退回合成，不能整條提示音掛掉。"""
    cfg = Config()
    cfg.tts.chime_source = "files"
    cfg.tts.chime_files = [str(tmp_path / "nope.mp3")]
    s = tts.make_chime(cfg)
    assert s.size > 0
    assert float(np.abs(s).max()) > 0.05


def test_make_chime_start_and_end_can_use_different_files(tmp_path):
    """開頭一聲、結尾另一聲（使用者：「一聲高一聲低…不要一樣，我會搞錯」）。"""
    from voice_dispatch import audio

    cfg = Config()
    sr = cfg.audio.samplerate
    hi = np.sin(2 * np.pi * 880.0 * np.arange(int(0.15 * sr)) / sr).astype(np.float32)
    lo = np.sin(2 * np.pi * 440.0 * np.arange(int(0.35 * sr)) / sr).astype(np.float32)
    p_hi, p_lo = tmp_path / "hi.wav", tmp_path / "lo.wav"
    audio.write_wav(str(p_hi), hi, sr)
    audio.write_wav(str(p_lo), lo, sr)
    cfg.tts.chime_source = "files"
    cfg.tts.chime_start_files = [str(p_hi)]
    cfg.tts.chime_end_files = [str(p_lo)]
    assert tts.make_chime(cfg, cue="start").size == hi.size
    assert tts.make_chime(cfg, cue="end").size == lo.size
    # 沒設專屬清單的那一邊 → 退回共用的 chime_files
    cfg.tts.chime_files = [str(p_lo)]
    cfg.tts.chime_start_files = []
    assert tts.make_chime(cfg, cue="start").size == lo.size


