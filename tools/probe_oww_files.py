"""用 OwwSpotter 離線掃描任意音檔：印出每檔的 peak / 過門檻幀數。

用途：驗證「daemon 是不是被自己的提示音／TTS 觸發」。
跑法（必須用 Hermes venv + repo 在 PYTHONPATH）：
  PY=~/.hermes/hermes-agent/venv/bin/python3
  PYTHONPATH=~/Data/Dev/python/hermes-voice-dispatch \
    $PY tools/probe_oww_files.py <audio1> <audio2> ...
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Data/Dev/python/hermes-voice-dispatch"))
from voice_dispatch.oww import OwwSpotter  # noqa: E402

MODEL = os.path.expanduser(
    "~/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx"
)
BLOCK = 1024


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16)


def scan(path: str) -> None:
    if not os.path.isfile(path):
        print(f"[skip] {path} 不存在")
        return
    pcm = decode(path)
    spot = OwwSpotter(model_path=MODEL, threshold=0.85,
                      relaxed_threshold=0.60, window_frames=6, relaxed_hits=2)
    peak = 0.0
    over_strong = 0
    over_relaxed = 0
    dur = len(pcm) / 16000.0
    for i in range(0, len(pcm) - BLOCK + 1, BLOCK):
        hit = spot.feed(pcm[i:i + BLOCK])  # 命中會 reset
        s = spot.latest_score
        peak = max(peak, s)
        if s >= 0.85:
            over_strong += 1
        if s >= 0.60:
            over_relaxed += 1
    print(f"{os.path.basename(path):50s} {dur:6.2f}s  peak={peak:.3f} "
          f">=0.85:{over_strong:3d}  >=0.60:{over_relaxed:3d}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        scan(p)
