"""tts.muted()：靜音測試模式——存在 mute 檔就完全不合成、不播放 TTS。"""
import os

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
