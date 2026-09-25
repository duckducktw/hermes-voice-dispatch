"""STT：正規化音檔（ffmpeg）+ 呼叫 stt_scoped.sh 取轉錄。

一律經由 stt_scoped.sh（cgroup 保護），不直接載入 faster-whisper，否則會 OOM。
腳本 stdout 為一行 JSON：{"transcript": "...", "duration_sec": ..., ...}，取 transcript。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from .config import Config


class SttError(RuntimeError):
    """STT 前處理或轉錄失敗。"""


def normalize_wav(
    src: str,
    dst: str,
    cfg: Config,
    logger=None,
) -> str:
    """用 ffmpeg 把任意音檔正規化成 16kHz 單聲道 wav。回傳 dst。"""
    samplerate = cfg.audio.samplerate
    argv = [
        cfg.stt.ffmpeg_bin, "-y",
        "-i", src,
        "-ac", "1",
        "-ar", str(samplerate),
        "-f", "wav",
        dst,
    ]
    try:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=cfg.stt.timeout_sec,
        )
    except FileNotFoundError as exc:
        raise SttError(f"找不到 ffmpeg（{cfg.stt.ffmpeg_bin}）：{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SttError(f"ffmpeg 正規化逾時：{exc}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace")
        if logger:
            logger.error("ffmpeg 失敗：%s", err)
        raise SttError(f"ffmpeg 正規化失敗（rc={proc.returncode}）：{err[-500:]}")
    if not Path(dst).exists():
        raise SttError(f"ffmpeg 未產生輸出檔：{dst}")
    return dst


def _parse_transcript(stdout: str) -> str:
    """從 stt_scoped.sh 的 stdout 解析 transcript。"""
    text = (stdout or "").strip()
    if not text:
        raise SttError("STT 沒有輸出")
    # 腳本輸出一行 JSON；保險起見取最後一個非空行
    last_line = ""
    for line in text.splitlines():
        if line.strip():
            last_line = line.strip()
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise SttError(f"無法解析 STT JSON：{exc}；原始輸出：{last_line[:300]}") from exc
    if "transcript" not in data:
        raise SttError(f"STT JSON 缺少 transcript 欄位：{last_line[:300]}")
    return str(data.get("transcript", "")).strip()


def transcribe_wav(wav_path: str, cfg: Config, logger=None, model: str = "") -> str:
    """呼叫 stt_scoped.sh <wav> - 取得轉錄字串。

    model 傳空字串時用 cfg.stt.model（再空則用腳本預設＝large-v3）；
    有值則覆寫，讓喚醒詞這種冷啟動路徑可以換成快模型。
    """
    script = cfg.resolved_scoped_script()
    if not os.path.exists(script):
        raise SttError(f"找不到 STT 腳本：{script}")

    argv = [script, wav_path, "-"]
    chosen = model or cfg.stt.model
    if chosen:
        argv.append(chosen)

    try:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=cfg.stt.timeout_sec,
        )
    except FileNotFoundError as exc:
        raise SttError(f"無法執行 STT 腳本：{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SttError(f"STT 逾時（{cfg.stt.timeout_sec}s）：{exc}") from exc

    stderr = (proc.stderr or b"").decode("utf-8", "replace")
    if proc.returncode != 0:
        if logger:
            logger.error("STT 腳本失敗（rc=%s）：%s", proc.returncode, stderr)
        raise SttError(f"STT 腳本失敗（rc={proc.returncode}）：{stderr[-500:]}")

    stdout = (proc.stdout or b"").decode("utf-8", "replace")
    if logger and stderr:
        logger.debug("STT stderr：%s", stderr[-500:])
    return _parse_transcript(stdout)


def transcribe_samples(
    samples,
    cfg: Config,
    tmpdir: Optional[str] = None,
    logger=None,
    model: str = "",
) -> str:
    """把 numpy 樣本寫成 wav 後轉錄。回傳轉錄字串。

    model 見 transcribe_wav：空字串＝用 cfg.stt.model / 腳本預設。
    """
    import tempfile

    from . import audio

    d = tmpdir or tempfile.gettempdir()
    raw_path = os.path.join(d, f"vd_stt_{os.getpid()}_raw.wav")
    norm_path = os.path.join(d, f"vd_stt_{os.getpid()}_norm.wav")
    try:
        audio.write_wav(raw_path, samples, cfg.audio.samplerate)
        # 已是 16k mono，但仍過一次 ffmpeg 確保格式一致
        normalize_wav(raw_path, norm_path, cfg, logger=logger)
        return transcribe_wav(norm_path, cfg, logger=logger, model=model)
    finally:
        for p in (raw_path, norm_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
