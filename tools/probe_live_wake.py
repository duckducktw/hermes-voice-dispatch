"""即時麥克風 OWW 探針：把每段高分音訊存成 wav，用來確認「到底是什麼在觸發喚醒」。

跑法：
  PY=~/.hermes/hermes-agent/venv/bin/python3
  PYTHONPATH=~/Data/Dev/python/hermes-voice-dispatch \
    $PY tools/probe_live_wake.py --seconds 120 --out /tmp/owwlive

行為：
  - 從 WordForum_USB 以 16k/mono 取音，逐 1024 block 餵 OwwSpotter。
  - 每 5 秒印一次峰值（>=0.05 才印），與 daemon 的「觀測」格式一致。
  - 峰值 >= --dump-threshold（預設 0.55）時，把「該幀往前 4 秒」的原始音訊
    存成 wav，方便事後聽／再分析。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Data/Dev/python/hermes-voice-dispatch"))
from voice_dispatch.oww import OwwSpotter  # noqa: E402

MODEL = os.path.expanduser("~/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx")
SR = 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default="/tmp/owwlive")
    ap.add_argument("--device", default="WordForum")
    ap.add_argument("--blocksize", type=int, default=1024)
    ap.add_argument("--dump-threshold", type=float, default=0.55)
    ap.add_argument("--tail-seconds", type=float, default=4.0)
    args = ap.parse_args()

    import sounddevice as sd
    import wave

    os.makedirs(args.out, exist_ok=True)
    keep = int(round(args.tail_seconds * SR / args.blocksize))
    ring: deque = deque(maxlen=keep)
    pre_len = int(args.tail_seconds * SR)
    tail = np.zeros(0, dtype=np.int16)

    spot = OwwSpotter(model_path=MODEL, threshold=0.85,
                      relaxed_threshold=0.60, window_frames=6, relaxed_hits=2)

    dev = None
    for i, d in enumerate(sd.query_devices()):
        if args.device.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            dev = i
            break
    print(f"[probe] device={dev} ({sd.query_devices(dev)['name'] if dev is not None else 'N/A'})"
          f" seconds={args.seconds} dump>={args.dump_threshold}")

    hits = []
    last = time.time()
    peak = 0.0
    win_relaxed = 0
    stop_at = time.time() + args.seconds

    def cb(indata, frames, t, status):  # noqa: ANN001
        nonlocal peak, last, win_relaxed, tail
        block = np.frombuffer(bytes(indata), dtype=np.int16).copy()
        ring.append(block)
        buf = np.concatenate(list(ring))
        tail = buf[-pre_len:]
        s = spot.feed(block)
        score = spot.latest_score
        if score > peak:
            peak = score
        win_relaxed = max(win_relaxed, spot.relaxed_count())
        if score >= args.dump_threshold:
            ts = time.strftime("%H%M%S")
            path = os.path.join(args.out, f"hit_{ts}_{score:.2f}.wav")
            j = 0
            while os.path.exists(path):
                j += 1
                path = os.path.join(args.out, f"hit_{ts}_{score:.2f}_{j}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(tail.tobytes())
            hits.append((ts, float(score), path))
            print(f"[HIT] {ts} score={score:.3f} -> {path}", flush=True)
        if time.time() - last >= 5.0:
            if peak >= 0.05:
                print(f"[obs] {time.strftime('%H:%M:%S')} 近5秒最高分 {peak:.3f} "
                      f"（弱命中 {win_relaxed} 幀）", flush=True)
            peak = 0.0
            win_relaxed = 0
            last = time.time()

    with sd.RawInputStream(samplerate=SR, blocksize=args.blocksize, channels=1,
                           dtype="int16", device=dev, callback=cb):
        while time.time() < stop_at:
            sd.sleep(200)

    print(f"[probe] 結束，共 {len(hits)} 筆 >= {args.dump_threshold}")
    for ts, sc, p in hits:
        print(f"  {ts}  {sc:.3f}  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
