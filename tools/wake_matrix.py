#!/usr/bin/env python3
"""喚醒詞「矩陣測試」：不同音色 / 語速 / 音量 / 變量（+間隔）→ 逐軸 FP/FN/延遲。

用法：
    python3 tools/wake_matrix.py --out /tmp/wake_matrix [--concurrency 8] [--axes voice,speed,gain,noise,gap]

輸出：終端機逐軸表格 + <out>/result.jsonl（每筆樣本的軸值、是否喚醒、延遲）。

設計：
- 基底＝edge-tts 合成（可快取在 <out>/base/），每筆再套單一軸的變換：
  * voice：不同的英文聲音（口音/性別）
  * speed：atempo 變速（保音高）
  * gain ：音量增減 dB
  * noise：疊加白噪音到指定 SNR
  * gap ：把「Hey」與「Hermes」分開合成再以不同間隔接起來
- 喚醒延遲＝（喚醒樣本位置 − 語音結束位置）/ 取樣率；負值＝在語音結束前就醒。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_dispatch import cascade  # noqa: E402
from voice_dispatch.config import Config  # noqa: E402

try:
    import edge_tts
except ImportError:  # pragma: no cover
    sys.exit("需要 edge-tts（在 ~/.hermes/hermes-agent/venv 內）")

SR = 16000
BLOCK = 1024

POS_TEXTS = [
    "Hey Hermes.",
    "Hey Hermes, restart the server.",
    "Okay, hey Hermes, do it",
]
NEG_TEXTS = [
    "Hey hermit", "Hey her miss", "hay her mess", "Hey her mouse",
    "Hey her mom", "Hey, harm us", "her mess", "hermes",
]
VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural",
    "en-AU-NatashaNeural", "en-IN-NeerjaNeural", "en-US-MichelleNeural", "en-IN-PrabhatNeural",
]
SPEEDS = [0.8, 1.0, 1.25]          # atempo（保音高）
GAINS_DB = [-20.0, -12.0, -6.0, 0.0]
SNRS_DB = [None, 15.0, 8.0, 3.0]   # None = 乾淨
GAPS_SEC = [0.15, 0.5, 0.9]        # 「Hey」「Hermes」之間的停頓


# ───────────────────────── 音訊工具 ─────────────────────────
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.clip(x, -1, 1).astype(np.int16).tobytes())


def apply_speed(x: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return x
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
         "-i", "pipe:0", "-filter:a", f"atempo={factor}", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=x.astype(np.float32).tobytes(), capture_output=True)
    if p.returncode != 0:
        return x
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def apply_gain(x: np.ndarray, db: float) -> np.ndarray:
    return x * (10.0 ** (db / 20.0))


def add_noise(x: np.ndarray, snr_db: float | None, seed: int = 0) -> np.ndarray:
    if snr_db is None:
        return x
    rng = np.random.default_rng(seed)
    speech_rms = float(np.sqrt(np.mean(x ** 2))) or 1e-6
    noise = rng.standard_normal(len(x)).astype(np.float32)
    # 輕度低通（房間噪音比較像），用簡單移動平均
    k = 4
    noise = np.convolve(noise, np.ones(k, dtype=np.float32) / k, mode="same")
    n_rms = float(np.sqrt(np.mean(noise ** 2))) or 1e-6
    noise *= (speech_rms / (10.0 ** (snr_db / 20.0))) / n_rms
    return (x + noise).astype(np.float32)


def speech_end(x: np.ndarray, frame: int = 320) -> int:
    """最後一個「有聲音」的位置（相對門檻＝最大 frame RMS 的 15%），跨音量穩健。"""
    if len(x) < frame:
        return 0
    idx = np.arange(0, len(x) - frame, frame)
    rms = np.array([float(np.sqrt(np.mean(x[i:i + frame] ** 2))) for i in idx])
    if rms.max() <= 0:
        return 0
    hit = np.where(rms > 0.15 * rms.max())[0]
    return int((hit[-1] + 1) * frame) if len(hit) else 0


def trim_silence(x: np.ndarray, frame: int = 160, rel: float = 0.08) -> np.ndarray:
    """去掉頭尾靜音（edge-tts 會在句尾補約 1 秒靜音，會污染「間隔」量測）。"""
    if len(x) < frame:
        return x
    idx = np.arange(0, len(x) - frame, frame)
    rms = np.array([float(np.sqrt(np.mean(x[i:i + frame] ** 2))) for i in idx])
    if rms.max() <= 0:
        return x
    keep = np.where(rms > rel * rms.max())[0]
    if not len(keep):
        return x
    return x[idx[keep[0]]: idx[keep[-1]] + frame]


# ───────────────────────── edge-tts 合成 ─────────────────────────
async def synth(text: str, voice: str, out: Path, sem: asyncio.Semaphore) -> bool:
    if out.exists():
        return True
    mp3 = out.with_suffix(".mp3")
    async with sem:
        for a in range(3):
            try:
                await edge_tts.Communicate(text, voice).save(str(mp3))
                break
            except Exception:
                if a == 2:
                    return False
                await asyncio.sleep(1.5 * (a + 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16", str(out)], capture_output=True)
    mp3.unlink(missing_ok=True)
    return r.returncode == 0 and out.exists()


# ───────────────────────── 主流程 ─────────────────────────
def evaluate(out: Path, axes: set[str], concurrency: int) -> dict:
    base = out / "base"
    base.mkdir(parents=True, exist_ok=True)
    cfg = Config(); cfg.audio.blocksize = BLOCK
    det = cascade.WakeCascade(cfg)

    # 1) 合成基底
    jobs = []
    for txt in POS_TEXTS + NEG_TEXTS:
        for v in VOICES:
            sid = f"{abs(hash((txt, v))) % (10**10):010d}"
            jobs.append((txt, v, base / f"{sid}.wav"))
    print(f"合成 {len(jobs)} 個基底…")
    asyncio.run(_synth_all(jobs, concurrency))
    jobs = [j for j in jobs if j[2].exists()]
    print(f"基底就緒 {len(jobs)}")

    results = []   # dict(axis, value, label, woke, lat)

    def run(x, axis, value, label, seed=0):
        det.reset(hard=True)
        # 補尾端靜音＝模擬真實「連續串流」（否則短檔在收訊窗還沒滿就結束，
        # 最後一次確認不會發生，會低估喚醒率）。
        stream = np.concatenate([x, np.zeros(int((cfg.wake.verify_window_sec + 0.5) * SR),
                                             dtype=np.float32)])
        woke_at = None
        for i in range(0, len(stream), BLOCK):
            if det.feed(stream[i:i + BLOCK]):
                woke_at = i + BLOCK
                break
        lat = None
        if woke_at is not None:
            lat = (woke_at - speech_end(x)) / SR
        results.append({"axis": axis, "value": value, "label": label,
                        "woke": woke_at is not None, "lat": lat})

    for txt, v, path in jobs:
        label = 1 if txt in POS_TEXTS else 0
        x0 = read_wav(path)
        if "voice" in axes:      # 音色軸＝每個聲音都算一筆（乾淨、正常速度/音量）
            run(x0, "voice", v, label)
        if "speed" in axes:
            for sp in SPEEDS:
                run(apply_speed(x0, sp), "speed", sp, label)
        if "gain" in axes:
            for g in GAINS_DB:
                run(apply_gain(x0, g), "gain", g, label)
        if "noise" in axes:
            for snr in SNRS_DB:
                run(add_noise(x0, snr, seed=len(results)), "noise",
                    "clean" if snr is None else f"{snr:g}dB", label)

    # gap 軸（分開合成 hey / hermes，去掉 TTS 補的靜音後，插入真正的停頓）
    if "gap" in axes:
        gapj = []
        for vi, v in enumerate(VOICES[:4]):
            gapj.append(("Hey,", v, base / f"_hey_{vi}.wav"))
            gapj.append(("Hermes", v, base / f"_hermes_{vi}.wav"))
        asyncio.run(_synth_all(gapj, concurrency))
        for vi, v in enumerate(VOICES[:4]):
            hw = base / f"_hey_{vi}.wav"; mw = base / f"_hermes_{vi}.wav"
            if not (hw.exists() and mw.exists()):
                continue
            h = trim_silence(read_wav(hw)); m = trim_silence(read_wav(mw))
            for gap in GAPS_SEC:
                x = np.concatenate([h, np.zeros(int(gap * SR), dtype=np.float32), m])
                run(x, "gap", f"{gap:g}s", 1)

    return {"results": results, "n_base": len(jobs)}


async def _synth_all(jobs, concurrency):
    sem = asyncio.Semaphore(concurrency)
    await asyncio.gather(*[synth(t, v, p, sem) for (t, v, p) in jobs])


def report(results: list[dict]) -> None:
    axes = []
    for r in results:
        if r["axis"] not in axes:
            axes.append(r["axis"])
    for axis in axes:
        rows = [r for r in results if r["axis"] == axis]
        vals = []
        for r in rows:
            if r["value"] not in vals:
                vals.append(r["value"])
        print(f"\n=== 軸：{axis} ===")
        print(f"{'條件':>10} | {'正樣本':>5} {'喚醒':>4} {'漏判':>4} {'漏判率':>6} | "
              f"{'負樣本':>5} {'誤判':>4} {'誤判率':>6} | {'延遲中位':>8} {'延遲max':>8}")
        for v in vals:
            rs = [r for r in rows if r["value"] == v]
            pos = [r for r in rs if r["label"] == 1]
            neg = [r for r in rs if r["label"] == 0]
            tp = sum(1 for r in pos if r["woke"])
            fn = len(pos) - tp
            fp = sum(1 for r in neg if r["woke"])
            lat = sorted(r["lat"] for r in pos if r["lat"] is not None)
            med = lat[len(lat) // 2] * 1000 if lat else 0.0
            mx = lat[-1] * 1000 if lat else 0.0
            fnr = f"{fn/len(pos):.0%}" if pos else "-"
            fpr = f"{fp/len(neg):.0%}" if neg else "-"
            print(f"{str(v):>10} | {len(pos):>5} {tp:>4} {fn:>4} {fnr:>6} | "
                  f"{len(neg):>5} {fp:>4} {fpr:>6} | {med:>7.0f}ms {mx:>7.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/wake_matrix")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--axes", default="voice,speed,gain,noise,gap")
    ap.add_argument("--json-out", default=None, help="把逐筆結果寫成 jsonl（預設 <out>/result.jsonl）")
    args = ap.parse_args()
    out = Path(args.out)
    axes = {a.strip() for a in args.axes.split(",") if a.strip()}
    res = evaluate(out, axes, args.concurrency)
    report(res["results"])
    jp = Path(args.json_out) if args.json_out else out / "result.jsonl"
    jp.parent.mkdir(parents=True, exist_ok=True)
    with jp.open("w", encoding="utf-8") as fh:
        for r in res["results"]:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n逐筆結果 → {jp}")


if __name__ == "__main__":
    main()
