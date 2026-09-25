#!/usr/bin/env python3
"""麥克風電平探針：校正拍手門檻、確認裝置真的有訊號。

用法（務必用 Hermes venv 的 python）：

    PY=~/.hermes/hermes-agent/venv/bin/python3
    $PY tools/probe_levels.py                 # 量環境底噪 10 秒
    $PY tools/probe_levels.py --seconds 5
    $PY tools/probe_levels.py --loopback      # 從喇叭放 1kHz，驗證麥克風收得到
    $PY tools/probe_levels.py --device 28     # 指定裝置（index 或名稱子字串）

為什麼需要這支：本機 PipeWire 的「預設來源」曾經指到一個收不到任何聲音的節點
（HiFi__Mic2__source），症狀是「守護程式在跑、但拍手永遠沒反應」。這支可以在
30 秒內分辨「是門檻太高」還是「這顆裝置根本是死的」。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd


def measure(device, seconds: float) -> dict:
    peak = 0.0
    rms_values = []
    with sd.InputStream(
        samplerate=16000, channels=1, blocksize=1024, dtype="float32", device=device
    ) as stream:
        start = time.time()
        while time.time() - start < seconds:
            data, _ = stream.read(1024)
            rms = float(np.sqrt((data[:, 0].astype(np.float64) ** 2).mean()))
            peak = max(peak, rms)
            rms_values.append(rms)
    arr = np.array(rms_values) if rms_values else np.zeros(1)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": peak,
        "blocks": len(rms_values),
    }


def resolved_name(device) -> str:
    try:
        if device is None:
            return f"(系統預設) {sd.query_devices(kind='input')['name']}"
        if isinstance(device, int):
            return f"[{device}] {sd.query_devices(device)['name']}"
        idx = sd.query_devices(device)["index"]
        return f"[{idx}] {sd.query_devices(device)['name']}"
    except Exception as exc:  # noqa: BLE001
        return f"(無法解析：{exc})"


def make_tone(path: str, seconds: float = 4.0) -> None:
    sr = 16000
    t = (0.8 * np.sin(2 * np.pi * 1000 * np.arange(int(seconds * sr)) / sr)).astype("float32")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", path],
        input=t.tobytes(), capture_output=True, check=True,
    )


def loopback(device) -> int:
    """喇叭放 1kHz，同時量麥克風。用來證明這顆裝置真的收得到聲音。"""
    tone = "/tmp/vd_probe_tone.wav"
    make_tone(tone)
    stop = [False]
    result = {}

    def record() -> None:
        result.update(measure(device, 6.0))

    def play() -> None:
        while not stop[0]:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tone],
                stdin=subprocess.DEVNULL,
            )

    th_play = threading.Thread(target=play, daemon=True)
    th_play.start()
    time.sleep(1.0)
    th_rec = threading.Thread(target=record)
    th_rec.start()
    th_rec.join()
    stop[0] = True
    time.sleep(0.3)
    print(f"播音中：max={result['max']:.5f}  median={result['median']:.5f}")
    if result["max"] > 0.05:
        print("✅ 這顆裝置收得到聲音（拍手 abs_floor=0.05 應該可用）")
        return 0
    print("❌ 幾乎收不到聲音 → 換裝置（--device）或檢查硬體靜音鍵/路由")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="麥克風電平探針")
    ap.add_argument("--device", default="HiFi__Mic1__source",
                    help="裝置 index 或名稱子字串（預設 HiFi__Mic1__source）")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--loopback", action="store_true", help="放 1kHz 驗證收得到聲音")
    ap.add_argument("--list", action="store_true", help="列出所有輸入裝置後結束")
    args = ap.parse_args()

    if args.list:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"[{i:2}] {d['name']}")
        return 0

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    print(f"裝置：{resolved_name(device)}")
    if args.loopback:
        return loopback(device)

    st = measure(device, args.seconds)
    print(f"{args.seconds:.0f} 秒：median={st['median']:.5f}  p95={st['p95']:.5f}  max={st['max']:.5f}")
    print(f"（clap.abs_floor 預設 0.05、vad.speech_rms_threshold 預設 0.02 可作對照）")
    if st["median"] < 1e-4 and st["max"] < 1e-3:
        print("⚠️ 訊號幾乎為零 → 這顆裝置可能是死的（3.5mm 耳麥孔沒插東西？）")
        print("   試 --loopback 驗證，或 --list 換一顆。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
