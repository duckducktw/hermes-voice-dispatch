"""TTS：用 numpy 合成 beep 提示音；用 edge-tts／Gemini TTS 合成語音並播放。

合成引擎由 `tts.engine` 決定：
  - "edge"（預設）＝ edge-tts（免費、快，但合成腔明顯）
  - "gemini"＝ Google Gemini TTS（更像真人，可用「風格指示」演出商業大佬口吻；
    免費層每 model 每天 10 次 → 會在多個 model 之間輪替，全掛才退回 edge-tts）

兩者都需要網路；在 --dry-run / 測試時不會呼叫網路（由呼叫端控制）。
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
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


def _synth_note(
    freq: float,
    dur_sec: float,
    samplerate: int,
    tts_cfg,
    seed: int = 0,
) -> np.ndarray:
    """合成「一個有質感的音」，回傳未正規化的 float 波形。

    為什麼不是純正弦：單一正弦 + 指數衰減聽起來就是「空洞的電子嗶」。
    為什麼也不是木頭/鼓（加了雜訊敲擊瞬態）：那會變成「火車／木魚」——
    使用者 2026-09-25 兩版都退貨：「我要蘋果那種感覺」。

    蘋果（iOS/macOS）那票提示音（Glass / Marimba / Tri-tone / Note）的核心是
    **FM 調變**，不是加法泛音也不是雜訊敲擊：
        y(t) = sin(2πft + I(t)·sin(2π·f·R·t))
      1. 載波 f 決定音高；調變比 R（非整數＝鐘／玻璃，整數＝木琴／馬林巴）
      2. 調變指數 I 在幾十毫秒內從大衰減到 0 → 起音瞬間明亮、之後只剩純淨的基頻
         （這個「亮度快速收乾」就是蘋果音「乾淨但有質感」的來源）
      3. 全程沒有雜訊：乾淨、帶一點殘響，聽起來貴。
    另可疊加法泛音列（partials）補厚度，但蘋果系通常只留基頻附近。
    """
    n = max(1, int(round(dur_sec * samplerate)))
    t = np.arange(n, dtype=np.float64) / samplerate
    wave = np.zeros(n, dtype=np.float64)
    base = max(float(getattr(tts_cfg, "chime_decay_sec", 0.28) or 0.28), 1e-4)

    fm_ratio = float(getattr(tts_cfg, "chime_fm_ratio", 0.0) or 0.0)
    if fm_ratio > 0:
        idx = float(getattr(tts_cfg, "chime_fm_index", 0.0) or 0.0)
        idx_decay = max(float(getattr(tts_cfg, "chime_fm_decay_sec", 0.05) or 0.05), 1e-4)
        idx_env = idx * np.exp(-t / idx_decay)
        wave += np.sin(2 * np.pi * freq * t + idx_env * np.sin(2 * np.pi * freq * fm_ratio * t))

    partials = list(getattr(tts_cfg, "chime_partials", []) or [])
    for ratio, amp, dmult in partials:
        ratio = float(ratio)
        if ratio <= 0:
            continue
        d = max(base * float(dmult), 1e-4)
        wave += float(amp) * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t / d)

    if not np.any(wave):
        return wave

    detune_cents = float(getattr(tts_cfg, "chime_detune_cents", 0.0) or 0.0)
    if detune_cents:
        # 第二層微失諧（拍頻）→ 厚度。振幅略低、衰減略快，不然會糊掉。
        factor = 2 ** (detune_cents / 1200.0) - 1.0
        for ratio, amp, dmult in partials:
            ratio = float(ratio)
            if ratio <= 0:
                continue
            d = max(base * float(dmult) * 1.06, 1e-4)
            wave += (
                0.55
                * float(amp)
                * np.sin(2 * np.pi * freq * ratio * (1.0 + factor) * t)
                * np.exp(-t / d)
            )

    peak = float(np.max(np.abs(wave)))
    if peak > 1e-9:
        wave /= peak
    attack_noise = float(getattr(tts_cfg, "chime_attack_noise", 0.0) or 0.0)
    if attack_noise > 0:
        # 敲擊瞬態（雜訊 burst）。**蘋果系音色不要開這個**（會變木頭／火車）。
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, n) * np.exp(-t / 0.005)
        npeak = float(np.max(np.abs(noise))) or 1.0
        w = attack_noise
        wave = (1.0 - w) * wave + w * (noise / npeak)
    # 3ms 淡入，避免開頭爆音
    fade = min(max(1, int(0.003 * samplerate)), n)
    wave[:fade] *= np.linspace(0.0, 1.0, fade)
    return wave


def _add_room(wave: np.ndarray, samplerate: int, taps, wet: float) -> np.ndarray:
    """加一點「空間感」（幾道衰減延遲 = 極簡殘響）。

    乾巴巴的短音聽起來像電腦提示音；一點點尾韻會立刻變得有質感。
    刻意只做幾道早期反射，不用真殘響（免費卷積太貴也沒必要）。
    """
    if wet <= 0 or not taps:
        return wave
    out = wave.copy()
    for delay_sec, gain in taps:
        k = int(round(float(delay_sec) * samplerate))
        if 0 < k < len(wave):
            out[k:] += float(wet) * float(gain) * wave[:-k]
    return out


_ASSET_CACHE: dict = {}


def _decode_audio(path: str, samplerate: int) -> np.ndarray:
    """用 ffmpeg 把任意格式（mp3/m4a/aiff…）解成單聲道 float32 @ samplerate。"""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", expand(path), "-f", "f32le",
         "-ac", "1", "-ar", str(samplerate), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 解碼失敗：{path}（{proc.stderr.decode('utf-8', 'replace')[:200]}）")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _finalise(out: np.ndarray, cfg: Config, samplerate: int) -> np.ndarray:
    """統一收尾：尾端 10ms 淡出（避免「啪」一聲）+ 正規化到設定音量。"""
    if out.size == 0:
        return out.astype(np.float32)
    fade = min(int(0.01 * samplerate), out.size)
    if fade > 0:
        out = out.copy()
        out[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = out * (float(cfg.tts.chime_volume) / peak)
    return out.astype(np.float32)


def _chime_from_files(cfg: Config, samplerate: int, cue: str = "start") -> Optional[np.ndarray]:
    """用 config 指定的音檔當提示音（原廠素材模式）。

    使用者 2026-09-25：「你找找蘋果素材」——不要再合成，直接用蘋果原廠音效。
    多個檔案會依序串接（之間插 `chime_gap_sec`），單一檔案＝整顆 cue 直接播。
    `cue="start"`（喚醒）與 `cue="end"`（聽完）可各掛不同音檔：
    `chime_start_files` / `chime_end_files`，沒設就退回共用的 `chime_files`。
    素材本身不放進 repo（版權），路徑由 config 指；檔案不存在就回 None 讓它退回合成。
    """
    paths = [p for p in _cue_paths(cfg, cue) if p and os.path.exists(p)]
    if not paths:
        return None
    max_n = int(round(float(getattr(cfg.tts, "chime_file_max_sec", 2.5) or 2.5) * samplerate))
    gap = np.zeros(max(0, int(round(float(cfg.tts.chime_gap_sec) * samplerate))), dtype=np.float32)
    chunks: List[np.ndarray] = []
    for i, p in enumerate(paths):
        key = (p, samplerate, os.path.getmtime(p))
        y = _ASSET_CACHE.get(key)
        if y is None:
            y = _decode_audio(p, samplerate)
            _ASSET_CACHE[key] = y
        if max_n > 0 and y.size > max_n:
            y = y[:max_n]
        chunks.append(y.astype(np.float32))
        if i != len(paths) - 1 and gap.size:
            chunks.append(gap)
    return np.concatenate(chunks) if chunks else None


def _cue_paths(cfg: Config, cue: str) -> List[str]:
    """取得某個 cue（start/end）要播的音檔清單。

    專屬清單（`chime_start_files` / `chime_end_files`）優先，
    沒設就退回共用的 `chime_files`。
    """
    specific = getattr(cfg.tts, f"chime_{cue}_files", None) or []
    chosen = specific or (getattr(cfg.tts, "chime_files", None) or [])
    return [expand(p) for p in chosen]


def make_chime(cfg: Config, samplerate: Optional[int] = None, cue: str = "start") -> np.ndarray:
    """產生提示音（float32 樣本）。`cue` 為 `"start"`（喚醒）或 `"end"`（聽完需求）。

    使用者 2026-09-25 定案流程：喚醒響一次、聽完需求再響一次，**且兩顆聽起來要不一樣**
    （「一聲高一聲低，像 Discord 開關 mic，但不要一樣，我會搞錯」）。
    音源有兩種，由 `tts.chime_source` 決定：
      - `"files"`：播 config 指定的**原廠／素材音檔**（優先）。
      - `"synth"`：用泛音列 + FM 調變 + 敲擊瞬態 + 殘響**合成**（參數全在 `tts.chime_*`）。
    合成是為了在沒有素材的機器上也能運作；素材模式才是有質感的正解。
    """
    sr = int(samplerate or cfg.audio.samplerate)
    source = str(getattr(cfg.tts, "chime_source", "synth") or "synth").lower()
    if source == "files":
        loaded = _chime_from_files(cfg, sr, cue)
        if loaded is not None and loaded.size:
            return _finalise(loaded, cfg, sr)
    return _make_chime_synth(cfg, sr)


def _make_chime_synth(cfg: Config, samplerate: int) -> np.ndarray:
    """合成版「咚咚」：FM 調變鈴聲（蘋果感）＋可選的加法泛音／敲擊瞬態／殘響。

    兩聲的間隔以「起音」計算；第一聲的尾韻會自然延伸進縫裡（像真的樂器）。
    """
    sr = int(samplerate)
    tts_cfg = cfg.tts
    n_note = max(1, int(round(float(tts_cfg.chime_tone_dur_sec) * sr)))
    n_gap = max(0, int(round(float(tts_cfg.chime_gap_sec) * sr)))
    n_tail = max(0, int(round(float(tts_cfg.chime_tail_sec) * sr)))
    note_len = n_note + n_tail
    taps = list(getattr(tts_cfg, "chime_reverb_taps", []) or [])
    wet = float(getattr(tts_cfg, "chime_reverb", 0.0) or 0.0)
    total = 2 * n_note + n_gap + n_tail
    out = np.zeros(total, dtype=np.float64)
    for i, freq in enumerate((tts_cfg.chime_tone1_hz, tts_cfg.chime_tone2_hz)):
        note = _synth_note(float(freq), note_len / float(sr), sr, tts_cfg, seed=1000 + i)
        note = _add_room(note, sr, taps, wet)
        start = i * (n_note + n_gap)
        end = min(total, start + note.size)
        out[start:end] += note[: end - start]
    return _finalise(out, cfg, sr)


def play_chime(cfg: Config, logger=None, cue: str = "start") -> bool:
    """產生並播放提示音（cue="start"／"end"）。靜音模式下不出聲（與 speak 一致）。"""
    if muted(cfg):
        if logger:
            logger.info("TTS 靜音中 → 跳過提示音")
        return False
    samples = make_chime(cfg, cue=cue)
    if samples.size == 0:
        return False
    path = os.path.join(tempfile.gettempdir(), f"vd_chime_{cue}_{os.getpid()}.wav")
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


def _gemini_api_key(cfg: Config) -> str:
    """取 Gemini API key：先看環境變數，再讀 ~/.hermes/.env。"""
    name = getattr(cfg.tts, "gemini_api_key_env", "GOOGLE_API_KEY") or "GOOGLE_API_KEY"
    if os.environ.get(name):
        return os.environ[name]
    path = expand(getattr(cfg.tts, "gemini_env_file", "~/.hermes/.env") or "")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def _pcm_rate_from_mime(mime: str, default: int = 24000) -> int:
    """從 mimeType 取裸 PCM 的取樣率，例如 "audio/L16;codec=pcm;rate=16000" → 16000。"""
    m = re.search(r"rate\s*=\s*(\d+)", mime or "")
    if not m:
        return default
    rate = int(m.group(1))
    return rate if 8000 <= rate <= 48000 else default


def _sniff_container(data: bytes) -> Optional[List[str]]:
    """嗅探常見容器 magic。回傳 ffmpeg 的輸入參數（讓它自己解），非容器回 None。

    ⚠️ 這個嗅探是「爆音守門」的關鍵：把已壓縮的位元組（MP3/OGG/FLAC）硬當成
    裸 PCM 餵給 ffmpeg，出來就是**全振幅白噪音**——就是使用者聽到的「唸到一半
    突然超大聲沙」。所以任何認得出來的容器一律走 -i 讓 ffmpeg 自行判斷。
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return []                      # WAV
    if data[:4] == b"OggS":
        return []                      # Ogg/Opus/Vorbis
    if data[:4] == b"fLaC":
        return []                      # FLAC
    if data[:3] == b"ID3":
        return []                      # MP3 with ID3
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return []                      # MPEG audio frame sync (裸 MP3/AAC-ADTS)
    if data[4:8] == b"ftyp":
        return []                      # MP4/M4A
    return None


def _gemini_write_audio(data: bytes, out_path: str, mime: str = "",
                        logger=None) -> None:
    """把 Gemini 回的音訊一律轉成 mp3。

    格式判斷順序（2026-09-27 修正 — 原本只看 RIFF 會造成中途爆音）：
      1. **容器嗅探**（WAV/Ogg/FLAC/MP3/MP4）→ 交給 ffmpeg 自己解。
      2. 認不出容器才當裸 PCM，且**取樣率一律從 mimeType 取**
         （`audio/L16;codec=pcm;rate=16000`），不再硬寫 24000。

    為什麼不能只信 RIFF：實測 gemini-3.8 回真 WAV、gemini-3.1 回裸 PCM 但
    mimeType 照樣寫 "audio/wav"。但**免費層每 model 每天只有 10 次配額**，
    用完會自動輪替到下一顆 model，而不同 model 回的取樣率／容器不一樣。
    舊碼把非 RIFF 的一切都當 24kHz s16le → 若實際是 16kHz 就變尖銳加速的
    雜音、若實際是壓縮容器就變**全振幅白噪音**。長文分多段合成時只有換到
    model 的那一段爆掉，聽起來就是「唸到一半突然超大聲沙」。
    """
    args = _sniff_container(data)
    if args is not None:
        src = out_path + ".src"
        args = ["-i", src]
        kind = "container"
    else:
        src = out_path + ".pcm"
        rate = _pcm_rate_from_mime(mime)
        args = ["-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", src]
        kind = f"raw pcm {rate}Hz"
    with open(src, "wb") as fh:
        fh.write(data)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args + [out_path],
                       check=True)
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    if logger:
        logger.info("TTS 解碼：%s（mime=%r，%d bytes）", kind, mime, len(data))


def _apply_speed(path: str, speed: float, logger=None) -> None:
    """用 ffmpeg atempo 精準調語速（**不變調**，音色不受影響）。

    為什麼不用「風格指示」控速度：實測在 prompt 裡寫「語速稍快」與「語速偏快」，
    Gemini 回傳的檔案**逐位元相同**（都 18860 bytes / 4.6s）——指示對速度幾乎沒有
    可調性。所以改成合成後用 atempo 後處理：`speed=1.15` ＝快 15%，是精準、可微調、
    且兩引擎（edge/gemini）都通用的旋鈕。

    atempo 單段有效範圍 0.5~2.0；超出就串接多段。
    """
    remaining = float(speed)
    if remaining <= 0 or abs(remaining - 1.0) < 0.01:
        return
    filters = []
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    tmp = path + ".speed.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-filter:a", ",".join(filters), tmp],
        check=True,
    )
    os.replace(tmp, path)
    if logger:
        logger.info("TTS 語速調整：x%.3f", speed)


def _duration(path: str) -> float:
    """用 ffprobe 量音檔長度（秒）。量不到回 0。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return 0.0


def _cps(path: str, n_chars: int) -> float:
    """回傳「字/秒」（量不到回 0）。用來抓 Gemini 把風格指示也唸出來的異常。"""
    dur = _duration(path)
    return (n_chars / dur) if (dur > 0.3 and n_chars > 0) else 0.0


def normalise_pace(path: str, text: str, target_cps: float, logger=None) -> None:
    """把語速**正規化**成「每秒 target_cps 個字」，靠 atempo 後處理。

    為什麼需要（2026-09-26 實測）：同一顆 voice 在不同 Gemini TTS model 上語速差很多——
    Charon 同一句話 3.1-flash-tts＝4.6s、3.8-flash-tts＝5.72s、3.8-flash-lite＝6.64s。
    而免費層每顆 model 只有 10 次/天，配額用完就會自動換顆 → 固定倍率（speed）會讓
    語速忽快忽慢。改成「先量實際秒數、再算出該補多少 atempo」，不論抽到哪顆都一樣快。

    只在字數足夠時動作（太短的量測誤差大）。
    """
    n = len([c for c in (text or "") if not c.isspace()])
    if n < 8:
        return
    dur = _duration(path)
    if dur <= 0.3:
        return
    # 方向別搞反：atempo 是「倍率」，要從 actual 字/秒 調到 target 字/秒
    #   → atempo = target / actual（>1＝加速）。實測曾寫成 actual/target → 越調越慢。
    actual_cps = n / dur
    atempo = float(target_cps) / actual_cps
    atempo = min(max(atempo, 0.7), 1.6)   # 保護：別把聲音拉壞
    if logger:
        logger.info("TTS 語速正規化：%.2f 字/秒 → 目標 %.2f（atempo x%.3f，原 %.2fs）",
                    actual_cps, target_cps, atempo, dur)
    _apply_speed(path, atempo, logger=None)


def _limit_peaks(path: str, logger=None) -> None:
    """把超過 0 dBFS 的過衝壓回來，避免播放端削波爆裂聲。

    為什麼需要（2026-09-27 實測）：Gemini 有些段落本身就貼著滿刻度（max 0.0 dBFS），
    再經 `atempo` 重新取樣 + mp3 重編碼，峰值會**衝過 0 dBFS**——實測 astats
    量到 `Peak level dB: +1.99`。超過滿刻度的樣本在播放時被硬截平，就是短促的
    「啪／沙」爆裂聲。這跟 `_is_noise_burst` 抓的「整段白噪音」是**不同的故障**：
    前者是瞬間削波（Flat factor 0、Abs Peak count 少），後者是整段訊號都壞掉。

    用 alimiter 做真峰限制。**注意**：只掛 alimiter 不夠——實測 +1.99 dB 的檔
    只降到 +0.84 dB，因為 mp3 有損重編碼本身會再產生新的過衝（解碼後的波形不等於
    編碼前）。所以先用 `volume` 靜態降 3 dB 給重編碼留餘裕，再用 alimiter 壓真峰。
    """
    tmp = path + ".lim.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             # -3dB 預降（給 mp3 重編碼的過衝留餘裕）→ 真峰限到 -3 dBFS
             "-af", "volume=-3dB,alimiter=limit=0.708:level=false",
             "-b:a", "128k", tmp],
            check=True)
    except Exception as exc:  # noqa: BLE001 - 限幅失敗不該擋住播放
        if logger:
            logger.warning("峰值限幅失敗（%s）→ 用原檔", exc)
        return
    os.replace(tmp, path)
    if logger:
        logger.info("TTS 峰值限幅：預降 3 dB + 真峰限至 -3 dBFS（避免播放削波）")


#: 爆音判準的門檻（見 `_verdict_noise`）。實測值，別憑感覺改。
NOISE_MAX_DBFS = -1.0     # 峰值頂到滿刻度才算可疑
NOISE_CREST_DB = 12.0     # 波峰因數低於此＝訊號「密實」＝雜訊而非語音


def _verdict_noise(vmax: float, vmean: float) -> bool:
    """純函式判準：給 max/mean dBFS，回答「這是不是爆音雜訊」。

    抽成純函式是為了能用**實測到的真實 dB 數值**做表格測試——合成訊號
    （sine + tremolo）無法代理真人語音的動態，實測 crest 只有 6~7 dB，
    會讓「不要誤殺大聲語音」的測試假性失敗。真人語音要靠實錄數值驗證。
    """
    return vmax >= NOISE_MAX_DBFS and (vmax - vmean) < NOISE_CREST_DB


def _volume_stats(path: str):
    """用 ffmpeg volumedetect 量 (max_dBFS, mean_dBFS)。量不到回 None。"""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", path, "-af", "volumedetect",
             "-f", "null", "-"],
            capture_output=True, text=True, check=True)
    except Exception:  # noqa: BLE001 - 量不到就不做判斷（寧可放過也別誤殺）
        return None
    err = proc.stderr or ""
    m_max = re.search(r"max_volume:\s*(-?[\d.]+) dB", err)
    m_mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", err)
    if not (m_max and m_mean):
        return None
    return float(m_max.group(1)), float(m_mean.group(1))


def _is_noise_burst(path: str, logger=None) -> bool:
    """判斷音檔是不是「全振幅白噪音」（解碼格式判錯的典型結果）。

    為什麼需要（2026-09-27 使用者回報「唸到一半突然超大聲沙」）：把壓縮位元組
    或錯取樣率的資料當裸 PCM 解，會得到接近滿刻度、且 RMS 貼著峰值的訊號。
    人聲相反——峰值高但 RMS 低（波峰因數大，因為字與字之間有安靜段落）。

    門檻怎麼定的（2026-09-27 實測 ffmpeg volumedetect）：
        MP3 當裸 PCM 解     max   0.0 / mean  -6.6 → crest  6.6  ← 真正的 bug
        白噪音 a=0.99       max   0.0 / mean  -9.3 → crest  9.3
        真人語音（原始）     max  -5.3 / mean -23.1 → crest 17.8  ✅ 不誤殺
        真人語音（推到滿刻度）max   0.0 / mean -14.3 → crest 14.3  ✅ 不誤殺
        正弦波              max -18.5 / mean -21.5 → crest  3.0（靠 max 排除）
    所以 crest 門檻取 12：夾在爆音 9.3 與真人語音 14.3 之間。
    **別調回 6.0** —— 實測爆音是 6.6，設 6.0 整個守門會失效（第一版的錯）。
    """
    stats = _volume_stats(path)
    if stats is None:
        return False
    vmax, vmean = stats
    noisy = _verdict_noise(vmax, vmean)
    if logger and noisy:
        logger.warning("偵測到爆音／雜訊：max=%.1f dB mean=%.1f dB crest=%.1f dB",
                       vmax, vmean, vmax - vmean)
    return noisy


def synth_gemini_to_file(text: str, out_path: str, cfg: Config, logger=None) -> str:
    """用 Google Gemini TTS 合成（比 edge-tts 更像真人，且能演出「商業大佬」口吻）。

    為什麼要換引擎（2026-09-26 使用者：「不要，我要更像真人，那種商業大佬的感覺」）：
    edge-tts 的合成腔到頂就是那樣，換聲音名救不了。

    ⚠️ Gemini 會把 prompt 整段唸出來，所以風格指示**必須**寫成分節標題的形式：

        # 風格指示
        <風格描述>
        # 台詞
        <真正要唸的字>

    實測：台詞 26 字 → 5.3s（正確，只唸台詞）；若寫成「請用…口吻說出：<台詞>」
    會被連指示一起唸成 14.1s。所以別改成自然語言開頭。

    免費層**每個 model 每天只有 10 次**配額 → 依序輪替 gemini_model /
    gemini_model_fallbacks，任一成功即用；全失敗才由 speak() 退回 edge-tts。
    """
    import base64
    import json as _json
    import urllib.request

    key = _gemini_api_key(cfg)
    if not key:
        raise RuntimeError("找不到 Gemini API key（tts.gemini_api_key_env / .env）")

    style = (getattr(cfg.tts, "gemini_style", "") or "").strip()
    prompt = f"# 風格指示\n{style}\n# 台詞\n{text}" if style else text
    voice = getattr(cfg.tts, "voice", "") or "Charon"
    timeout = float(getattr(cfg.tts, "gemini_timeout_sec", 60) or 60)
    models = [getattr(cfg.tts, "gemini_model", "")] + list(
        getattr(cfg.tts, "gemini_model_fallbacks", []) or []
    )
    last_err: Optional[Exception] = None
    # 每個 model 最多試兩種 prompt：先帶風格指示；若發現它把指示也唸出來，就退回純台詞。
    attempts = [(True, prompt)] if style else []
    attempts.append((False, text))
    min_cps = float(getattr(cfg.tts, "gemini_min_cps", 3.0) or 3.0)
    n_chars = len([c for c in text if not c.isspace()])
    for model in [m for m in models if m]:
        for with_style, body_text in attempts:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            body = {
                "contents": [{"parts": [{"text": body_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
                },
            }
            req = urllib.request.Request(
                url, data=_json.dumps(body).encode(), headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    payload = _json.load(resp)
                _inline = payload["candidates"][0]["content"]["parts"][0]["inlineData"]
                data = base64.b64decode(_inline["data"])
                mime = _inline.get("mimeType", "") or ""
            except Exception as exc:  # noqa: BLE001 - 配額/網路/格式 → 換下一顆 model
                last_err = exc
                break
            _gemini_write_audio(data, out_path, mime=mime, logger=logger)
            # 爆音守門：解碼錯把壓縮位元組當裸 PCM 會變全振幅白噪音（使用者回報
            # 「唸到一半突然超大聲沙」）。用峰值／RMS 抓，壞了就換下一顆 model。
            if _is_noise_burst(out_path, logger=logger):
                last_err = RuntimeError(f"解碼出雜訊（mime={mime!r}）")
                if logger:
                    logger.warning("Gemini TTS 輸出疑似雜訊（mime=%r）→ 換下一顆 model", mime)
                break
            # 品質守門：Gemini 偶爾會把「# 風格指示」也當台詞唸出來。
            # 實測 26 字的句子正常約 4.6~6.6s；唸出指示會變成 15.3s（字/秒 掉到 1.7）。
            cps = _cps(out_path, n_chars)
            if with_style and n_chars >= 8 and 0 < cps < min_cps:
                if logger:
                    logger.warning(
                        "Gemini 疑似連風格指示一起唸（%.2f 字/秒 < %.2f）→ 去掉指示重合成",
                        cps, min_cps)
                continue
            if logger:
                logger.info("TTS 合成（gemini %s，voice=%s，%d bytes，%.2f 字/秒%s）",
                            model, voice, len(data), cps,
                            "" if with_style else "，無風格指示")
            return out_path
    raise RuntimeError(f"Gemini TTS 全部 model 都失敗：{last_err}")


def _mute_path(cfg: Config) -> str:
    return expand(getattr(cfg.tts, "mute_file", "") or "")


def muted(cfg: Config) -> bool:
    """是否處於「靜音測試」模式。

    存在 mute 檔就完全不合成、不播放——使用者要邊測邊不吵，而且改狀態不用重啟。
    """
    p = _mute_path(cfg)
    return bool(p) and os.path.exists(p)


# ── 長文分段（2026-09-27 使用者：「完整的訊息不會唸出來」）──────────────────
# 使用者要求**完整唸完**，所以超過 `tts.speak_chunk_chars` 就切成幾段分別合成、
# 再無縫接起來（實測單次 1200 字仍完整，但整段報告可到 2000 字，分段才穩）：
#   - 優先切在句尾（。！？；!?;）
#   - 單句本身就超長 → 硬切
#   - chunk_chars <= 0 ＝不分段（維持單次合成）
DEFAULT_SPEAK_CHUNK_CHARS = 240


def split_speech_chunks(text: str, limit: int = DEFAULT_SPEAK_CHUNK_CHARS) -> List[str]:
    """把長文切成每段 <= limit 字（盡量切在句尾）。limit <= 0 ＝不分段。"""
    t = (text or "").strip()
    if not t:
        return []
    if limit <= 0 or len(t) <= limit:
        return [t]
    out: List[str] = []
    buf = ""
    for s in re.split(r"(?<=[。！？；!?;])\s*", t):
        s = s.strip()
        if not s:
            continue
        if len(s) > limit:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(s[i:i + limit] for i in range(0, len(s), limit))
            continue
        if len(buf) + len(s) <= limit:
            buf += s
        else:
            if buf:
                out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    return out or [t]


def _concat_audio(parts: List[str], out_path: str, logger=None) -> None:
    """把多個音檔依序接成一個（重新編碼，避免各段串流參數不一致）。"""
    listfile = out_path + ".concat.txt"
    with open(listfile, "w", encoding="utf-8") as fh:
        for p in parts:
            fh.write("file '%s'\n" % p.replace("'", "'\\''"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", listfile, "-c:a", "libmp3lame", "-ar", "24000", "-ac", "1", out_path],
            check=True)
    finally:
        try:
            os.remove(listfile)
        except OSError:
            pass
    if logger:
        logger.info("TTS 分段合成：%d 段 → 合併成一個檔", len(parts))


#: Gemini 的 prebuilt 聲音名（`tts.voice` 用這些時，edge-tts 不認得）。
GEMINI_VOICES = frozenset({
    "Charon", "Orus", "Alnilam", "Algenib", "Puck", "Kore", "Fenrir", "Aoede",
    "Leda", "Zephyr", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina",
    "Erinome", "Rasalgethi", "Laomedeia", "Achernar", "Achird", "Sadachbia",
    "Schedar", "Gacrux", "Pulcherrima", "Zubenelgenubi", "Vindemiatrix",
    "Sadaltager", "Sulafat", "Callirrhoe", "Autonoe",
})

#: `tts.voice` 是 Gemini 聲音名時，退回 edge-tts 要改用的預設聲音。
DEFAULT_EDGE_VOICE = "zh-TW-YunJheNeural"


def _edge_voice(cfg: Config) -> str:
    """挑 edge-tts 能用的聲音名。

    為什麼需要（2026-09-27 實測）：`tts.voice` 兩引擎共用，設 gemini 時填的是
    Gemini prebuilt 名（例如 "Charon"）。Gemini 配額用完退回 edge-tts 時，
    edge_tts 會直接 `ValueError: Invalid voice 'Charon'` 把整段合成打掉——
    **備援路徑等於不存在**。所以要在這裡換成 edge 的聲音名。

    優先序：`tts.edge_voice`（明確指定）→ `tts.voice`（若不是 Gemini 名）→ 內建預設。
    """
    explicit = (getattr(cfg.tts, "edge_voice", "") or "").strip()
    if explicit:
        return explicit
    voice = (getattr(cfg.tts, "voice", "") or "").strip()
    if voice and voice not in GEMINI_VOICES and "-" in voice:
        return voice          # 看起來是 edge 的 locale 式聲音名（zh-TW-XxxNeural）
    return DEFAULT_EDGE_VOICE


def _synth_one(text: str, out_path: str, cfg: Config, engine: str, logger=None) -> None:
    """合成單一段落（gemini 失敗自動退回 edge-tts）。"""
    if engine == "gemini":
        try:
            synth_gemini_to_file(text, out_path, cfg, logger=logger)
            return
        except Exception as exc:  # noqa: BLE001 - 退回 edge-tts
            if logger:
                logger.warning("Gemini TTS 失敗（%s）→ 退回 edge-tts", exc)
    voice = _edge_voice(cfg)
    if logger:
        logger.info("TTS 合成（edge-tts，voice=%s）", voice)
    synth_to_file(text, out_path, voice, getattr(cfg.tts, "rate", ""))


def _apply_pace(path: str, text: str, cfg: Config, logger=None) -> None:
    """語速：優先「正規化」（跨 model 一致）；沒設才用固定倍率 speed。"""
    cps = float(getattr(cfg.tts, "speak_cps", 0.0) or 0.0)
    if cps > 0:
        normalise_pace(path, text, cps, logger=logger)
    else:
        _apply_speed(path, float(getattr(cfg.tts, "speed", 1.0) or 1.0), logger=logger)


def _split_in_half_at_sentence(text: str):
    """在接近中點處切兩半（優先句尾，其次逗號／空白，最後硬切）。"""
    mid = len(text) // 2
    for sep in ("。", "！", "？", "；", "，", ",", " "):
        i = text.rfind(sep, 0, mid + 1)
        if i >= mid // 2:
            return text[:i + 1].strip(), text[i + 1:].strip()
    return text[:mid].strip(), text[mid:].strip()


def _synth_unit(text: str, out_path: str, cfg: Config, engine: str,
                logger=None, depth: int = 0) -> None:
    """合成「一段」台詞；若疑似被模型截斷就自動再切半重合成（遞迴上限 2 層）。

    為什麼要這個守門（2026-09-27 實測）：Gemini TTS **單次輸出有音訊長度上限**，
    超過就一直截短——同一顆 model 下：

        200 字 → 57.7s（4.94 字/秒，正常）
       1200 字 → 150.4s（7.56 字/秒，已偏快）
       2000 字 → 104.7s（18.09 字/秒，**根本不可能**＝明顯被截斷、後半段沒唸）

    真實語速約 4~5 字/秒，所以合成後「字/秒」遠高於目標＝模型把內容吃掉了。
    靠這個訊號自動半切重合成，比單純調小分段門檻更 robust（不用猜上限）。
    """
    _synth_one(text, out_path, cfg, engine, logger=logger)
    tar = float(getattr(cfg.tts, "speak_cps", 0.0) or 0.0)
    n = len([c for c in text if not c.isspace()])
    if depth >= 2 or tar <= 0 or n < 40:
        return
    dur = _duration(out_path)
    if dur <= 0.3:
        return
    cps = n / dur
    if cps <= tar * 2.2:
        return
    a, b = _split_in_half_at_sentence(text)
    if not b:
        return
    if logger:
        logger.warning("TTS 疑似被模型截斷（%.2f 字/秒 ≫ 目標 %.2f，%d 字）→ 切半重合成",
                       cps, tar, n)
    pa, pb = out_path + ".a.mp3", out_path + ".b.mp3"
    try:
        _synth_unit(a, pa, cfg, engine, logger=logger, depth=depth + 1)
        _synth_unit(b, pb, cfg, engine, logger=logger, depth=depth + 1)
        _concat_audio([pa, pb], out_path, logger=logger)
    finally:
        for p in (pa, pb):
            try:
                os.remove(p)
            except OSError:
                pass


def speak(text: str, cfg: Config, logger=None) -> bool:
    """合成語音並播放。失敗時記 log 但不丟例外（語音提示非關鍵路徑）。

    長文（> `tts.speak_chunk_chars`）會分句分段合成再接起來 → **完整唸完不截斷**。
    """
    if not text:
        return False
    if muted(cfg):
        if logger:
            logger.info("TTS 靜音中 → 跳過：%r", text)
        return False
    base = os.path.join(tempfile.gettempdir(), f"vd_tts_{os.getpid()}")
    path = base + ".mp3"
    engine = (getattr(cfg.tts, "engine", "edge") or "edge").strip().lower()
    chunk_chars = int(getattr(cfg.tts, "speak_chunk_chars",
                             DEFAULT_SPEAK_CHUNK_CHARS) or 0)
    chunks = split_speech_chunks(text, chunk_chars)
    parts: List[str] = []
    try:
        if len(chunks) <= 1:
            _synth_unit(text, path, cfg, engine, logger=logger)
            _apply_pace(path, text, cfg, logger=logger)
        else:
            if logger:
                logger.info("TTS 長文（%d 字）→ 切成 %d 段合成", len(text), len(chunks))
            for i, ch in enumerate(chunks):
                p = f"{base}.part{i}.mp3"
                _synth_unit(ch, p, cfg, engine, logger=logger)
                _apply_pace(p, ch, cfg, logger=logger)
                parts.append(p)
            _concat_audio(parts, path, logger=logger)
    except Exception as exc:  # noqa: BLE001 - 網路/合成失敗都不該讓主流程崩潰
        if logger:
            logger.warning("TTS 合成失敗：%s", exc)
        return False
    finally:
        for p in parts:
            try:
                os.remove(p)
            except OSError:
                pass
    # 播放前做兩件事（2026-09-27 使用者回報「還是會」爆音後補上）：
    #   1. 真峰限幅 → 擋掉 atempo/mp3 重編碼造成的過衝削波（實測 +1.99 dBFS）。
    #   2. 留下電平診斷 → 下次真的爆音時，log 直接有現場數據可比對，不用重現。
    _limit_peaks(path, logger=logger)
    if logger:
        stats = _volume_stats(path)
        if stats:
            vmax, vmean = stats
            logger.info("TTS 播放前電平：max=%.1f dB mean=%.1f dB crest=%.1f dB（%.1fs）%s",
                        vmax, vmean, vmax - vmean, _duration(path),
                        "  ⚠️疑似雜訊" if _verdict_noise(vmax, vmean) else "")
    try:
        return audio.play_file(path, cfg.audio, logger=logger)
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
