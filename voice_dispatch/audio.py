"""音訊 I/O：裝置列舉、InputStream 迴圈、WAV 讀寫、播放。

sounddevice 需要系統的 PortAudio 原生函式庫；若缺少會在 import 時丟 OSError。
因此這裡一律「延遲載入」（在函式內 import），讓其餘模組即使在沒有音效裝置的
環境也能被 import 與單元測試。缺裝置時給明確錯誤，不讓整支程式崩潰。
"""

from __future__ import annotations

import shlex
import subprocess
import wave
from contextlib import contextmanager
from typing import Iterator, List, Optional

import numpy as np

from .config import AudioConfig


class AudioUnavailable(RuntimeError):
    """麥克風 / 音效裝置或 PortAudio 不可用。"""


def _import_sounddevice():
    try:
        import sounddevice as sd  # 延遲載入
        return sd
    except Exception as exc:  # ImportError / OSError(PortAudio not found)
        raise AudioUnavailable(
            f"無法載入 sounddevice（可能缺少 PortAudio 或麥克風）：{exc}"
        ) from exc


# --------------------------------------------------------------------------
# 裝置列舉
# --------------------------------------------------------------------------
def list_input_devices() -> List[dict]:
    """回傳輸入裝置清單（sounddevice 格式）。無法載入時丟 AudioUnavailable。"""
    sd = _import_sounddevice()
    devices = sd.query_devices()
    result = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            result.append({"index": idx, **dev})
    return result


def describe_device(device) -> str:
    """把設定裡的輸入裝置解析成人類可讀字串（給啟動 log 用）。

    裝置指錯時症狀是「程式在跑但永遠沒反應」，有這行才查得動。
    """
    try:
        sd = _import_sounddevice()
        if device is None:
            info = sd.query_devices(kind="input")
            return f"(系統預設) {info['name']}"
        info = sd.query_devices(device)
        return f"[{info['index']}] {info['name']}"
    except AudioUnavailable as exc:
        return f"(無法解析：{exc})"
    except Exception as exc:  # noqa: BLE001
        return f"(無法解析 {device!r}：{exc})"


def format_device_list() -> str:
    """把輸入裝置整理成人類可讀字串。

    若 sounddevice 不可用，退化為呼叫 `arecord -l`（Linux）以便至少列出硬體。
    """
    try:
        devices = list_input_devices()
    except AudioUnavailable as exc:
        fallback = _arecord_fallback()
        msg = [f"[警告] {exc}"]
        if fallback:
            msg.append("改用 `arecord -l` 列出的擷取裝置：")
            msg.append(fallback)
        else:
            msg.append("且 `arecord -l` 也無法取得裝置。")
        return "\n".join(msg)

    if not devices:
        return "找不到任何輸入（麥克風）裝置。"

    lines = ["可用輸入（麥克風）裝置："]
    for dev in devices:
        lines.append(
            f"  [{dev['index']}] {dev['name']} "
            f"（輸入聲道 {dev['max_input_channels']}, "
            f"預設取樣率 {int(dev.get('default_samplerate', 0))} Hz）"
        )
    return "\n".join(lines)


def _arecord_fallback() -> Optional[str]:
    try:
        out = subprocess.run(
            ["arecord", "-l"],
            capture_output=True, text=True, timeout=5,
        )
        text = (out.stdout or "").strip()
        return text or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# 串流讀取
# --------------------------------------------------------------------------
@contextmanager
def input_stream(cfg: AudioConfig):
    """開一個 InputStream 的 context manager，保證離開時關閉（避免資源洩漏）。"""
    sd = _import_sounddevice()
    stream = sd.InputStream(
        samplerate=cfg.samplerate,
        channels=cfg.channels,
        blocksize=cfg.blocksize,
        dtype="float32",
        device=cfg.device,
    )
    stream.start()
    try:
        yield stream
    finally:
        try:
            stream.stop()
        finally:
            stream.close()


def read_blocks(stream, blocksize: int) -> Iterator[np.ndarray]:
    """從已開啟的 stream 持續讀取單聲道 float32 區塊。"""
    while True:
        data, _overflowed = stream.read(blocksize)
        # data shape = (frames, channels)；取第一聲道攤平成 1D
        yield np.asarray(data, dtype=np.float32)[:, 0].copy()


def record_seconds(cfg: AudioConfig, seconds: float) -> np.ndarray:
    """錄固定秒數並回傳 1D float32 樣本。用於喚醒詞視窗與回述確認。"""
    sd = _import_sounddevice()
    frames = int(round(seconds * cfg.samplerate))
    if frames <= 0:
        return np.zeros(0, dtype=np.float32)
    rec = sd.rec(
        frames,
        samplerate=cfg.samplerate,
        channels=cfg.channels,
        dtype="float32",
        device=cfg.device,
    )
    sd.wait()
    return np.asarray(rec, dtype=np.float32)[:, 0].copy()


# --------------------------------------------------------------------------
# WAV 讀寫（用 stdlib wave，不引入 soundfile）
# --------------------------------------------------------------------------
def write_wav(path: str, samples: np.ndarray, samplerate: int) -> None:
    """把 float32 [-1, 1] 樣本寫成 16-bit PCM 單聲道 wav。"""
    arr = np.asarray(samples, dtype=np.float32).ravel()
    # 夾限避免溢位，再轉 int16
    clipped = np.clip(arr, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """讀 wav 回傳 (float32 樣本, samplerate)。主要給測試用。"""
    with wave.open(path, "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        raw = wf.readframes(n)
    if width != 2:
        raise ValueError(f"只支援 16-bit wav，實得 {width * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    return data, sr


# --------------------------------------------------------------------------
# 播放
# --------------------------------------------------------------------------
def play_file(path: str, cfg: AudioConfig, logger=None) -> bool:
    """依 cfg.players 順序嘗試播放檔案。成功回傳 True。"""
    for tmpl in cfg.players:
        # 先用 shlex 拆成 argv，再把 {file} 佔位符換成實際路徑（保證含空白路徑也正確）
        argv = [part.replace("{file}", path) for part in shlex.split(tmpl)]
        if not argv:
            continue
        try:
            proc = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            if proc.returncode == 0:
                return True
            if logger:
                logger.debug("播放器 %s 失敗（rc=%s）：%s",
                             argv[0], proc.returncode,
                             (proc.stderr or b"").decode("utf-8", "replace"))
        except FileNotFoundError:
            continue  # 這個播放器沒安裝，試下一個
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.debug("播放器 %s 例外：%s", argv[0] if argv else "?", exc)
            continue
    if logger:
        logger.warning("所有播放器都無法播放：%s", path)
    return False


# --------------------------------------------------------------------------
# 擷取路徑健檢（死訊號偵測）
# --------------------------------------------------------------------------
def measure_level(cfg: AudioConfig, seconds: float = 2.0) -> Optional[dict]:
    """開 stream 讀 seconds 秒，回傳 {rms, peak, crest, samples}；失敗回 None。

    crest = peak/rms 是「死訊號 vs 活音訊」最有效的判準（不依賴當下有沒有人在出聲）：
    - 凍結/壞掉的擷取路徑吐近乎常數的直流 → crest ≈ 1.0
    - 真實音訊（連底噪）crest 至少 2，真人講話 5~8
    """
    try:
        sd = _import_sounddevice()
    except AudioUnavailable:
        return None
    chunks = []
    try:
        with sd.InputStream(
            samplerate=cfg.samplerate,
            channels=cfg.channels,
            blocksize=cfg.blocksize,
            dtype="float32",
            device=cfg.device,
        ) as stream:
            need = int(seconds * cfg.samplerate)
            got = 0
            # 丟掉開檔瞬間的暫態（常見一根大尖波），否則 crest 會爆高
            for _ in range(3):
                stream.read(cfg.blocksize)
            while got < need:
                data, _overflowed = stream.read(cfg.blocksize)
                arr = np.asarray(data, dtype=np.float64)[:, 0]
                chunks.append(arr)
                got += arr.size
    except Exception:  # noqa: BLE001 - 裝置忙碌/開不起來都不該讓主流程崩潰
        return None

    d = np.concatenate(chunks) if chunks else np.zeros(0)
    if d.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(d))))
    peak = float(np.max(np.abs(d)))
    crest = peak / rms if rms > 1e-12 else float("inf")
    return {"rms": rms, "peak": peak, "crest": crest, "samples": int(d.size)}


def level_verdict(m: Optional[dict]) -> str:
    """把 measure_level 的結果翻成可行動的結論。"""
    if m is None:
        return "無法量測（裝置開不起來／被佔用）"
    if m["crest"] < 2.0:
        return "死訊號（凍結的直流）→ 路由/驅動問題，調門檻沒用"
    if m["peak"] < 0.01:
        return "串流活著但幾乎收不到聲音（沒插麥克風／硬體靜音／增益過低）"
    return "有真實聲音"
