#!/usr/bin/env python3
"""產生喚醒詞誤判測試語料：一堆「近似發音」的正/負樣本（edge-tts 合成）。

用途：量測 Vosk KWS 對「hey hermes」的**誤判率（false positive）**與漏判率
（false negative），並作為兩段式（低功耗偵測 + 高模型確認）調門檻的基準。

用法：
    python3 tools/build_wake_corpus.py --out /tmp/wake_corpus [--concurrency 6]

輸出：
    <out>/*.wav            16 kHz / mono / s16（給 Vosk 吃）
    <out>/labels.jsonl     每行 {"file","label","text","voice","rate"}

label=1 ＝ 應該喚醒（真的在講 hey hermes 及其變體）
label=0 ＝ 不該喚醒（近似音／同音詞／一般句子）

⚠️ 這是**合成語料**，不能取代真人錄音；但足以當「回歸基準」——任何改動後
重跑，只要 FP/FN 沒有退步就安全。真人資料另存（見 README 的 calibration 章）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:  # pragma: no cover
    sys.exit("需要 edge-tts（在 ~/.hermes/hermes-agent/venv 內）")

# ── 應該喚醒的正樣本 ────────────────────────────────────────────────
POS_PHRASES = [
    "Hey Hermes.",
    "Hey, Hermes.",
    "Hey Hermes!",
    "Hey Hermes, restart the server.",
    "Hey Hermes, what time is it",
    "Okay, hey Hermes, do it",
    "Hey Hermes, are you there",
    "Hey Hermes, turn on the light",
]

# ── 不該喚醒的負樣本（近似發音＝最危險的一群）────────────────────────
NEG_PHRASES = [
    # 單獨的 hermes（使用者定案：只喊 hermes 不算）
    "hermes",
    "Hermes",
    # her + 各種 m 開頭詞（her mess/moss/mass/... 是主要誤判來源）
    "her mess",
    "her mess is",
    "her messy",
    "her moss",
    "her mass",
    "her mouse",
    "her my",
    "her miss",
    "her me",
    "her mace",
    "her mercy",
    "her met",
    "her mom",
    "her money",
    "her mail",
    "her mood",
    "her magic",
    "her manager",
    "her mattress",
    "her meal",
    "her memory",
    "her message",
    "her mother",
    "her guide",
    # 「hey + herX」＝最像的假陽性
    "Hey, her mess",
    "Hey her mouse",
    "Hey her mom",
    "Hey, harm us",
    "Hey her miss",
    "Hey hermit",
    # 其他近音
    "hermit",
    "a hermit",
    "hermitage",
    "hurries",
    "hurry mass",
    "Her Majesty",
    "hair mess",
    "hair moss",
    "hay her mess",
    "air mess",
    "her guess",
    # 一般句子（控制組）
    "Good morning everyone",
    "The weather is nice today",
    "Turn on the light please",
    "What time is it now",
    "Play some music",
    "Hey there, how are you",
    "Hey, listen to me",
]

# ── 英文聲音（多口音＝模擬不同人的發音）──────────────────────────────
VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural", "en-US-EricNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural", "en-IN-NeerjaNeural",
    "en-US-MichelleNeural", "en-IN-PrabhatNeural", "en-US-ChristopherNeural", "en-GB-LibbyNeural",
]
POS_VOICES = VOICES[:8]
NEG_VOICES = VOICES

# 語速變化（同一個人快慢會影響 Vosk 判定）
RATES = ["+0%", "-15%", "+15%"]


def _slug(text: str, idx: int) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:40]
    return f"{idx:04d}_{s or 'x'}"


async def _synth_one(text: str, voice: str, rate: str, wav: Path, sem: asyncio.Semaphore) -> bool:
    mp3 = wav.with_suffix(".mp3")
    async with sem:
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(text, voice, rate=rate)
                await comm.save(str(mp3))
                break
            except Exception as exc:  # 網路/服務偶發
                if attempt == 2:
                    print(f"  ! 合成失敗 {text!r} {voice}: {exc}", file=sys.stderr)
                    return False
                await asyncio.sleep(1.5 * (attempt + 1))
    # 轉 16k mono wav
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(wav)],
        capture_output=True,
    )
    mp3.unlink(missing_ok=True)
    return r.returncode == 0 and wav.exists()


async def build(out: Path, concurrency: int = 6) -> int:
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    jobs = []          # (text, voice, rate, path, label)
    idx = 0
    for text in POS_PHRASES:
        for v in POS_VOICES:
            rate = RATES[idx % len(RATES)]
            idx += 1
            jobs.append((text, v, rate, out / f"{_slug(text, idx)}.wav", 1))
    for text in NEG_PHRASES:
        for v in NEG_VOICES:
            rate = RATES[idx % len(RATES)]
            idx += 1
            jobs.append((text, v, rate, out / f"{_slug(text, idx)}.wav", 0))

    print(f"合成 {len(jobs)} 個樣本（並行 {concurrency}）…")
    tasks = [_synth_one(t, v, r, p, sem) for (t, v, r, p, _l) in jobs]
    results = await asyncio.gather(*tasks)

    labels_path = out / "labels.jsonl"
    ok = 0
    with labels_path.open("w", encoding="utf-8") as fh:
        for (text, v, rate, p, label), good in zip(jobs, results):
            if not good:
                continue
            fh.write(json.dumps(
                {"file": p.name, "label": label, "text": text, "voice": v, "rate": rate},
                ensure_ascii=False) + "\n")
            ok += 1
    print(f"完成：{ok}/{len(jobs)} → {labels_path}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/wake_corpus")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    n = asyncio.run(build(Path(args.out), args.concurrency))
    sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
