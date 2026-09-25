"""產生「低沉＋快＋抖動」提示音素材（可重現，全部合成，無版權問題）。

用途
----
hermes-voice-dispatch 的提示音（`tts.chime_start_files` / `tts.chime_end_files`）
可以用這支腳本重新產生，不需要外部素材。

設計（2026-09-25 使用者定案）
-----------------------------
- 語意：收到喚醒＝「懂咚」（低→高）；錄音結束或沒收到錄音＝反過來「咚懂」（高→低）。
- 音色：**低沉、短、抖動**（使用者：「低沉 快」＋前面說的「抖動」）。
- 技術要點（前幾版被退貨後歸納）：
    1. 純正弦＝空洞；加雜訊敲擊＝像火車 → 用「多泛音 + 拍頻 + 低通」堆厚度。
    2. 筆電喇叭 <150Hz 幾乎不可聞（前一個 subagent 實測）
       → 基頻取 131–247Hz，另疊 f0/2 的 sub 層提供「沉」但**不靠它才聽得到**。
    3. 抖動＝振幅調變（12–28Hz）＋微失諧第二層的拍頻。
    4. 短：單聲 0.13–0.22s 衰減，整組 0.32–0.50s。

用法
----
    ~/.hermes/hermes-agent/venv/bin/python3 tools/make_cues.py            # 產生全部到 assets/chime/
    ... tools/make_cues.py --variant 3 --loud --out /tmp/x                # 只做第 3 組並加大
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
from scipy.signal import butter, filtfilt

SR = 48000
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    b, a = butter(2, cutoff / (SR / 2), btype="low")
    return filtfilt(b, a, x)


def note(f0, dur, decay, trem_hz, trem_d, sub=0.55, click=0.10, cut=2000.0, seed=0):
    """一顆「低沉、短、會抖」的撞擊音。"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.exp(-t / decay)
    x = np.sin(2 * np.pi * f0 * t)
    x += 0.50 * np.sin(2 * np.pi * 2 * f0 * t) * np.exp(-t / (decay * 0.60))
    x += 0.22 * np.sin(2 * np.pi * 3 * f0 * t) * np.exp(-t / (decay * 0.40))
    # 微失諧第二層 → 拍頻（抖動感之一）
    x += 0.60 * np.sin(2 * np.pi * (f0 + 3.0) * t) * np.exp(-t / (decay * 1.10))
    if sub:   # 沉，但不靠它才聽得到
        x += sub * np.sin(2 * np.pi * (f0 / 2) * t) * np.exp(-t / (decay * 1.30))
    if click:  # 起音那一下（低通過，避免刺耳）
        rng = np.random.default_rng(seed)
        x += click * lowpass(rng.normal(0, 1, n), 700.0)
    x *= 1.0 - trem_d + trem_d * np.sin(2 * np.pi * trem_hz * t)   # 抖動
    x *= env
    x = lowpass(x, cut)
    a = int(0.002 * SR)
    x[:a] *= np.linspace(0, 1, a)
    f = int(0.008 * SR)
    x[-f:] *= np.linspace(1, 0, f)
    return (x / (float(np.abs(x).max()) or 1.0) * 0.72).astype(np.float32)


def cue(n1: np.ndarray, n2: np.ndarray, gap: float = 0.06) -> np.ndarray:
    g = np.zeros(int(gap * SR), dtype=np.float32)
    y = np.concatenate([n1, g, n2])
    return (y * (0.5 / (float(np.abs(y).max()) or 1.0))).astype(np.float32)


# 四組變體（1＝預設用來聽的基準口味）
VARIANTS = {
    1: dict(name="01_低快抖", lo=165.0, hi=233.0, dur=0.22, decay=0.070, trem_hz=22.0, trem_d=0.35),
    2: dict(name="02_更低更快", lo=147.0, hi=208.0, dur=0.18, decay=0.055, trem_hz=26.0, trem_d=0.40),
    3: dict(name="03_極低沉", lo=131.0, hi=196.0, dur=0.20, decay=0.065, trem_hz=20.0,
            trem_d=0.38, sub=0.75, cut=1500.0),
    4: dict(name="04_短促無尾", lo=175.0, hi=247.0, dur=0.13, decay=0.040, trem_hz=28.0,
            trem_d=0.45, sub=0.35, click=0.16, cut=2600.0),
}
# 使用者 2026-09-25 選定：第 3 組並要求「大聲點」→ 這組是預設交付口味
CHOSEN = 3
LOUD = "acompressor=threshold=-18dB:ratio=4:attack=3:release=60,loudnorm=I=-9:TP=-0.8:LRA=6"


def write_wav(path: str, x: np.ndarray) -> None:
    sys.path.insert(0, REPO)
    from voice_dispatch import audio
    audio.write_wav(path, x, SR)


def encode(x: np.ndarray, path: str, loud: bool) -> float:
    wav = "/tmp/_mk_cue.wav"
    write_wav(wav, x)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", wav]
    if loud:
        cmd += ["-af", LOUD]
    cmd += ["-b:a", "192k", path]
    subprocess.run(cmd, check=True)
    return x.size / SR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "assets", "chime"))
    ap.add_argument("--variant", type=int, default=None, help="只產生某一組（1-4）")
    ap.add_argument("--loud", action="store_true", help="加大音量（壓縮＋loudnorm）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    picks = [args.variant] if args.variant else sorted(VARIANTS)
    for i in picks:
        kw = dict(VARIANTS[i])
        name = kw.pop("name")
        lo_f, hi_f = kw.pop("lo"), kw.pop("hi")
        lo = note(lo_f, seed=1, **kw)
        hi = note(hi_f, seed=2, **kw)
        suffix = "_LOUD" if args.loud else ""
        # start＝懂咚（低→高）；end＝咚懂（高→低）
        s = cue(lo, hi)
        e = cue(hi, lo)
        p1 = os.path.join(args.out, f"{name}{suffix}_start_DONGDONG.mp3")
        p2 = os.path.join(args.out, f"{name}{suffix}_end_DONGDONG.mp3")
        encode(s, p1, args.loud)
        encode(e, p2, args.loud)
        print(f"{name}{suffix}: {s.size/SR:.2f}s -> {p1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
