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


def _gemini_write_audio(data: bytes, out_path: str) -> None:
    """Gemini 有時回 WAV、有時回裸 PCM（24kHz s16le mono）→ 一律轉成 mp3。

    只信 RIFF magic：實測 gemini-3.8 回真 WAV，gemini-3.1 回裸 PCM 但
    mimeType 照樣寫 "audio/wav"（信 header 會踩雷，ffmpeg 直接 Invalid data）。
    """
    if data[:4] == b"RIFF":
        src, args = out_path + ".wav", ["-i", out_path + ".wav"]
    else:
        src = out_path + ".pcm"
        args = ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", out_path + ".pcm"]
    with open(src, "wb") as fh:
        fh.write(data)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args + [out_path], check=True)
    finally:
        try:
            os.remove(src)
        except OSError:
            pass


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
    for model in [m for m in models if m]:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
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
            data = base64.b64decode(
                payload["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            )
        except Exception as exc:  # noqa: BLE001 - 換下一個 model（配額/網路/格式）
            last_err = exc
            continue
        _gemini_write_audio(data, out_path)
        if logger:
            logger.info("TTS 合成（gemini %s，voice=%s，%d bytes）", model, voice, len(data))
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


def speak(text: str, cfg: Config, logger=None) -> bool:
    """合成語音並播放。失敗時記 log 但不丟例外（語音提示非關鍵路徑）。"""
    if not text:
        return False
    if muted(cfg):
        if logger:
            logger.info("TTS 靜音中 → 跳過：%r", text)
        return False
    path = os.path.join(tempfile.gettempdir(), f"vd_tts_{os.getpid()}.mp3")
    engine = (getattr(cfg.tts, "engine", "edge") or "edge").strip().lower()
    try:
        if engine == "gemini":
            try:
                synth_gemini_to_file(text, path, cfg, logger=logger)
            except Exception as exc:  # noqa: BLE001 - 退回 edge-tts，語音提示不該中斷流程
                if logger:
                    logger.warning("Gemini TTS 失敗（%s）→ 退回 edge-tts", exc)
                synth_to_file(text, path, cfg.tts.voice, getattr(cfg.tts, "rate", ""))
        else:
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
