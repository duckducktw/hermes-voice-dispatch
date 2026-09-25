"""TTS：用 numpy 合成 beep 提示音；用 edge-tts 合成語音並播放。

edge-tts 需要網路；在 --dry-run / 測試時不會呼叫網路（由呼叫端控制）。
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import numpy as np

from . import audio
from .config import Config


def make_beep(
    freq: float,
    dur_sec: float,
    samplerate: int,
    volume: float = 0.3,
) -> np.ndarray:
    """合成一段正弦波 beep，回傳 float32 樣本（含淡入淡出避免爆音）。"""
    n = int(round(dur_sec * samplerate))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / samplerate
    wave = np.sin(2 * np.pi * freq * t) * volume
    # 5ms 淡入淡出
    fade = max(1, int(0.005 * samplerate))
    if 2 * fade < n:
        ramp = np.linspace(0.0, 1.0, fade)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave.astype(np.float32)


def play_beep(cfg: Config, logger=None) -> bool:
    """合成並播放提示音。"""
    samples = make_beep(
        cfg.tts.beep_freq, cfg.tts.beep_dur_sec,
        cfg.audio.samplerate, cfg.tts.beep_volume,
    )
    if samples.size == 0:
        return False
    path = os.path.join(tempfile.gettempdir(), f"vd_beep_{os.getpid()}.wav")
    try:
        audio.write_wav(path, samples, cfg.audio.samplerate)
        return audio.play_file(path, cfg.audio, logger=logger)
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


async def _synth_async(text: str, voice: str, out_path: str) -> None:
    import edge_tts  # 延遲載入

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synth_to_file(text: str, out_path: str, voice: str) -> str:
    """用 edge-tts 把文字合成成 mp3 檔。回傳 out_path。"""
    asyncio.run(_synth_async(text, voice, out_path))
    return out_path


def speak(text: str, cfg: Config, logger=None) -> bool:
    """合成語音並播放。失敗時記 log 但不丟例外（語音提示非關鍵路徑）。"""
    if not text:
        return False
    path = os.path.join(tempfile.gettempdir(), f"vd_tts_{os.getpid()}.mp3")
    try:
        synth_to_file(text, path, cfg.tts.voice)
    except Exception as exc:  # noqa: BLE001 - 網路/合成失敗都不該讓主流程崩潰
        if logger:
            logger.warning("TTS 合成失敗：%s", exc)
        return False
    try:
        return audio.play_file(path, cfg.audio, logger=logger)
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
