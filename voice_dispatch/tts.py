"""TTS：用 numpy 合成 beep 提示音；用 edge-tts 合成語音並播放。

edge-tts 需要網路；在 --dry-run / 測試時不會呼叫網路（由呼叫端控制）。
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import List, Optional

import numpy as np

from . import audio
from .config import Config, expand


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


def make_chime(cfg: Config, samplerate: Optional[int] = None) -> np.ndarray:
    """合成「咚咚」兩聲提示音（下行雙音，回傳 float32 樣本）。

    使用者要求（2026-09-25）：互動不要講話，只要「一個咚咚」。
    每聲用指數衰減包絡 + 4ms 淡入，聽起來像門鈴／提示音而不是刺耳的嗶。
    所有頻率／長度／音量都取自 config，不寫死。
    """
    sr = int(samplerate or cfg.audio.samplerate)
    common = max(1, int(0.004 * sr))          # 4ms 淡入，避免爆音
    parts: List[np.ndarray] = []
    tones = (cfg.tts.chime_tone1_hz, cfg.tts.chime_tone2_hz)
    dur = float(cfg.tts.chime_tone_dur_sec)
    for i, freq in enumerate(tones):
        n = int(round(dur * sr))
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float64) / sr
        env = np.exp(-t / max(dur * 0.45, 1e-6))
        wave = np.sin(2 * np.pi * freq * t) * env * float(cfg.tts.chime_volume)
        fade = min(common, n)
        wave[:fade] *= np.linspace(0.0, 1.0, fade)
        parts.append(wave.astype(np.float32))
        if i == 0:                             # 兩聲之間的縫
            gap_n = int(round(float(cfg.tts.chime_gap_sec) * sr))
            if gap_n > 0:
                parts.append(np.zeros(gap_n, dtype=np.float32))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def play_chime(cfg: Config, logger=None) -> bool:
    """合成並播放「咚咚」。靜音模式下不出聲（與 speak 一致）。"""
    if muted(cfg):
        if logger:
            logger.info("TTS 靜音中 → 跳過咚咚")
        return False
    samples = make_chime(cfg)
    if samples.size == 0:
        return False
    path = os.path.join(tempfile.gettempdir(), f"vd_chime_{os.getpid()}.wav")
    try:
        audio.write_wav(path, samples, cfg.audio.samplerate)
        return audio.play_file(path, cfg.audio, logger=logger)
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


async def _synth_async(text: str, voice: str, out_path: str, rate: str = "") -> None:
    import edge_tts  # 延遲載入

    communicate = edge_tts.Communicate(text, voice, rate=(rate or "+0%"))
    await communicate.save(out_path)


def synth_to_file(text: str, out_path: str, voice: str, rate: str = "") -> str:
    """用 edge-tts 把文字合成成 mp3 檔。回傳 out_path。

    rate 是 edge-tts 的語速（例如 "+30%"）——使用者要求「說話快一點」。
    """
    asyncio.run(_synth_async(text, voice, out_path, rate))
    return out_path


def _mute_path(cfg: Config) -> str:
    return expand(getattr(cfg.tts, "mute_file", "") or "")


def muted(cfg: Config) -> bool:
    """是否處於「靜音測試」模式。

    存在 mute 檔就完全不合成、不播放——使用者要邊測邊不吵，而且改狀態不用重啟。
    """
    p = _mute_path(cfg)
    return bool(p) and os.path.exists(p)


def speak(text: str, cfg: Config, logger=None) -> bool:
    """合成語音並播放。失敗時記 log 但不丟例外（語音提示非關鍵路徑）。"""
    if not text:
        return False
    if muted(cfg):
        if logger:
            logger.info("TTS 靜音中 → 跳過：%r", text)
        return False
    path = os.path.join(tempfile.gettempdir(), f"vd_tts_{os.getpid()}.mp3")
    try:
        synth_to_file(text, path, cfg.tts.voice, getattr(cfg.tts, "rate", ""))
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
