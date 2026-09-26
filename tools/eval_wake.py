#!/usr/bin/env python3
"""評測喚醒詞的誤判（false positive）、漏判（false negative）與延遲。

拿 `tools/build_wake_corpus.py` 產的語料，評測 Vosk、openWakeWord 或既有串接式
hey → hermes 路徑。輸出混淆矩陣、逐句誤判清單，以及相對語音結束的平均喚醒延遲。

用法：
    # 現行 baseline（保留相容的 --cascade）
    python3 tools/eval_wake.py --corpus /tmp/wake_corpus --cascade \
        --stage1-model ~/.local/share/hermes-voice-dispatch/vosk-model-small-en-us-0.15

    # openWakeWord 模型和門檻掃描
    python3 tools/eval_wake.py --corpus /tmp/wake_corpus --engine oww \
        --oww-threshold-sweep 0.3,0.5,0.6,0.7,0.8

    # 單一組設定
    python3 tools/eval_wake.py --corpus /tmp/wake_corpus --engine vosk \
        --stage1-model ~/.local/share/hermes-voice-dispatch/vosk-model-small-en-us-0.15 \
        --stage1-conf 0.9

    # 第一階段門檻掃描（看 FP/FN 的取捨）
    python3 tools/eval_wake.py --corpus /tmp/wake_corpus --stage1-conf-sweep 0.5,0.7,0.8,0.9,0.95,1.0

    # 兩段式：第一階段低門檻取候選 → 第二階段高模型 + 高門檻確認
    python3 tools/eval_wake.py --corpus /tmp/wake_corpus \
        --stage1-conf 0.6 \
        --stage2-model ~/.local/share/hermes-voice-dispatch/vosk-model-en-us-0.22-lgraph \
        --stage2-conf 0.9 --verify-window-sec 2.0

⚠️ 這裡的「覺醒」定義＝整個檔案跑完，任一時點命中（與 daemon 的常開行為一致）。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice_dispatch import kws  # noqa: E402
from voice_dispatch import cascade  # noqa: E402
from voice_dispatch import oww  # noqa: E402
from voice_dispatch.config import Config  # noqa: E402

SR = 16000
BLOCK = 1024


def _speech_end(samples: np.ndarray, frame: int = 320, thr: float = 0.02) -> int:
    """最後一個「有聲音」的樣本索引（能量 VAD）；用來量喚醒延遲。"""
    end = 0
    for i in range(0, max(0, len(samples) - frame), frame):
        if float(np.sqrt(np.mean(samples[i:i + frame] ** 2))) > thr:
            end = i + frame
    return end


def _run_cascade(corpus: Path, cfg: Config) -> dict:
    """用 cascade.WakeCascade 跑整個語料，量測 FP/FN 與喚醒延遲。"""
    rows = [json.loads(l) for l in (corpus / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    det = cascade.WakeCascade(cfg)
    tp = fn = fp = tn = 0
    fp_details = collections.Counter()
    fn_details = collections.Counter()
    lat = []
    for r in rows:
        x = _load_wav(corpus / r["file"])
        det.reset(hard=True)
        # 補尾端靜音＝模擬連續串流（見 wake_matrix.py 同註解）
        stream = np.concatenate([x, np.zeros(int((cfg.wake.verify_window_sec + 0.5) * SR),
                                             dtype=np.float32)])
        woke_at = None
        for i in range(0, len(stream), BLOCK):
            blk = stream[i:i + BLOCK]
            if det.feed(blk):
                woke_at = i + len(blk)
                break
        woke = woke_at is not None
        if r["label"] == 1:
            if woke:
                tp += 1
                lat.append((woke_at - _speech_end(x)) / SR)
            else:
                fn += 1
                fn_details[r["text"]] += 1
        else:
            if woke:
                fp += 1
                fp_details[r["text"]] += 1
            else:
                tn += 1
    return {
        "mode": "cascade", "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fp_details": fp_details, "fn_details": fn_details,
        "n_pos": sum(1 for r in rows if r["label"] == 1),
        "n_neg": sum(1 for r in rows if r["label"] == 0),
        "lat": lat,
    }


def _run_oww(
    corpus: Path,
    model_path: str,
    threshold: float,
    confirmation_frames: int,
    vad_threshold: float = 0.0,
) -> dict:
    """以 OwwSpotter 跑整個語料。每個 wav 都重置成獨立的一次喚醒嘗試。"""
    rows = [
        json.loads(line)
        for line in (corpus / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    try:
        det = oww.OwwSpotter(
            model_path,
            threshold,
            confirmation_frames,
            vad_threshold=vad_threshold,
        )
    except oww.OwwUnavailable as exc:
        raise RuntimeError(f"openWakeWord 無法評測：{exc}") from exc

    tp = fn = fp = tn = 0
    fp_details = collections.Counter()
    fn_details = collections.Counter()
    lat = []
    for row in rows:
        samples = _load_wav(corpus / row["file"])
        det.reset()
        woke_at = None
        # 尾端補一個 block；音檔最後不足 80ms 時也要讓 chunker 有機會完成一幀。
        stream = np.concatenate([samples, np.zeros(BLOCK, dtype=np.float32)])
        for i in range(0, len(stream), BLOCK):
            block = stream[i:i + BLOCK]
            if det.feed(block):
                woke_at = i + len(block)
                break
        woke = woke_at is not None
        if row["label"] == 1:
            if woke:
                tp += 1
                lat.append((woke_at - _speech_end(samples)) / SR)
            else:
                fn += 1
                fn_details[row["text"]] += 1
        elif woke:
            fp += 1
            fp_details[row["text"]] += 1
        else:
            tn += 1
    return {
        "mode": "oww",
        "threshold": threshold,
        "confirmation_frames": confirmation_frames,
        "vad_threshold": vad_threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "fp_details": fp_details,
        "fn_details": fn_details,
        "n_pos": sum(1 for row in rows if row["label"] == 1),
        "n_neg": sum(1 for row in rows if row["label"] == 0),
        "lat": lat,
    }


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR, f"{path} 非 {SR}Hz"
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _eval_file(
    samples: np.ndarray,
    stage1: kws.VoskSpotter,
    stage2,
    window_blocks: int,
    stage1_conf: float,
):
    """回傳 (stage1 最高候選信心度, stage2 是否確認通過, stage2 累計秒數)。

    stage1 用 min_conf=0 建（呼叫端負責），這裡只要出現候選就記 conf。
    stage2（kws.WakeVerifier）對候選當下的滾動窗做全詞彙確認。
    """
    stage1._rec.Reset()
    ring = collections.deque(maxlen=max(1, window_blocks))
    best_conf = None
    confirmed = None
    stage2_secs = 0.0
    woke_at = None
    for i in range(0, len(samples), BLOCK):
        blk = samples[i:i + BLOCK]
        hit = stage1.feed(blk)
        ring.append(blk)
        if hit is not None:
            if best_conf is None or stage1.last_conf > best_conf:
                best_conf = stage1.last_conf
            if stage2 is not None and confirmed is None:
                window = np.concatenate(ring)
                t0 = time.perf_counter()
                confirmed = stage2.verify(window)
                stage2_secs += time.perf_counter() - t0
            if stage1.last_conf >= stage1_conf and (
                stage2 is None or bool(confirmed)
            ):
                woke_at = i + len(blk)
                break
    return best_conf, confirmed, stage2_secs, woke_at


def _run(corpus: Path, stage1_model: str, stage1_conf: float,
         stage2_model: str | None, stage2_conf: float, window_sec: float,
         prefixes=None, variants=None, stage1_words=None) -> dict:
    rows = [json.loads(l) for l in (corpus / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    cfg = Config()
    prefixes = prefixes if prefixes is not None else cfg.wake.verify_prefixes
    variants = variants if variants is not None else cfg.wake.verify_variants
    stage1_words = stage1_words or ["hey hermes"]
    stage1 = kws.VoskSpotter(stage1_model, words=stage1_words, min_conf=0.0)
    stage2 = None
    if stage2_model:
        stage2 = kws.WakeVerifier(stage2_model, prefixes=prefixes, variants=variants,
                                  min_conf=stage2_conf)
    window_blocks = int(round(window_sec * SR / BLOCK))

    tp = fn = fp = tn = 0
    fp_details = collections.Counter()
    fn_details = collections.Counter()
    stage2_calls = 0
    stage2_secs = 0.0
    lat = []
    for r in rows:
        samples = _load_wav(corpus / r["file"])
        # Vosk 的 final result 有時需等尾端靜音才吐出，模擬常開串流補 0.5 秒。
        stream = np.concatenate([samples, np.zeros(int(0.5 * SR), dtype=np.float32)])
        best_conf, confirmed, s2, woke_at = _eval_file(
            stream, stage1, stage2, window_blocks, stage1_conf
        )
        if confirmed is not None:
            stage2_calls += 1
            stage2_secs += s2
        woke = best_conf is not None and best_conf >= stage1_conf
        if woke and stage2 is not None:
            woke = bool(confirmed)
        if r["label"] == 1:
            if woke:
                tp += 1
                if woke_at is not None:
                    lat.append((woke_at - _speech_end(samples)) / SR)
            else:
                fn += 1
                fn_details[r["text"]] += 1
        else:
            if woke:
                fp += 1
                fp_details[r["text"]] += 1
            else:
                tn += 1
    return {
        "stage1_conf": stage1_conf, "stage2": bool(stage2),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fp_details": fp_details, "fn_details": fn_details,
        "n_pos": sum(1 for r in rows if r["label"] == 1),
        "n_neg": sum(1 for r in rows if r["label"] == 0),
        "stage2_calls": stage2_calls,
        "stage2_avg_ms": (stage2_secs / stage2_calls * 1000) if stage2_calls else 0.0,
        "lat": lat,
    }


def _print(result: dict, verbose: bool = True) -> None:
    n_pos, n_neg = result["n_pos"], result["n_neg"]
    fp, fn = result["fp"], result["fn"]
    if result.get("mode") in {"cascade", "oww"}:
        lat = result.get("lat") or []
        average = sum(lat) / len(lat) if lat else 0.0
        med = sorted(lat)[len(lat) // 2] if lat else 0.0
        if result["mode"] == "cascade":
            tag = "串接 hey→hermes"
        else:
            tag = (
                "openWakeWord "
                f"threshold={result['threshold']:.2f} "
                f"confirm={result['confirmation_frames']} "
                f"vad={result['vad_threshold']:.2f}"
            )
        print(f"\n[{tag}]  正樣本 {n_pos} / 負樣本 {n_neg}")
        print(f"  喚醒成功 TP={result['tp']:3d}  漏判 FN={fn:3d} (漏判率 {fn/n_pos:.1%})")
        print(f"  誤判   FP={fp:3d}  正確略過 TN={result['tn']:3d} (誤判率 {fp/n_neg:.1%})")
        if lat:
            print(f"  平均喚醒延遲（相對語音結束）：{average*1000:.0f} ms"
                  f"；中位數 {med*1000:.0f} ms"
                  f"（min {min(lat)*1000:.0f} / max {max(lat)*1000:.0f} ms）")
        else:
            print("  平均喚醒延遲：（無樣本）")
        if verbose and result["fp_details"]:
            print("  誤判來源 top：")
            for t, c in result["fp_details"].most_common(12):
                print(f"    {c:3d}× {t!r}")
        if verbose and result["fn_details"]:
            print("  漏判來源：")
            for t, c in result["fn_details"].most_common(12):
                print(f"    {c:3d}× {t!r}")
        return
    tag = f"stage1_conf={result['stage1_conf']:.2f}"
    if result["stage2"]:
        tag += " + stage2(全詞彙)"
    print(f"\n[{tag}]  正樣本 {n_pos} / 負樣本 {n_neg}")
    print(f"  喚醒成功 TP={result['tp']:3d}  漏判 FN={fn:3d} (漏判率 {fn/n_pos:.1%})")
    print(f"  誤判   FP={fp:3d}  正確略過 TN={result['tn']:3d} (誤判率 {fp/n_neg:.1%})")
    lat = result.get("lat") or []
    if lat:
        print(f"  平均喚醒延遲（相對語音結束）：{sum(lat) / len(lat) * 1000:.0f} ms")
    else:
        print("  平均喚醒延遲：（無樣本）")
    if result["stage2"]:
        print(f"  stage2：呼叫 {result['stage2_calls']} 次，平均 {result['stage2_avg_ms']:.0f} ms/次")
    if verbose and result["fp_details"]:
        print("  誤判來源 top：")
        for t, c in result["fp_details"].most_common(12):
            print(f"    {c:3d}× {t!r}")
    if verbose and result["fn_details"]:
        print("  漏判來源：")
        for t, c in result["fn_details"].most_common(12):
            print(f"    {c:3d}× {t!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="/tmp/wake_corpus")
    ap.add_argument("--engine", choices=("vosk", "oww", "cascade"), default="vosk",
                    help="要評測的引擎（預設 vosk；--cascade 仍保留相容）")
    ap.add_argument("--stage1-model",
                    default="~/.local/share/hermes-voice-dispatch/vosk-model-small-en-us-0.15")
    ap.add_argument("--stage1-words", default=None,
                    help="逗號分隔；第一階段（限制詞彙）的詞彙表（預設 hey hermes）")
    ap.add_argument("--stage1-conf", type=float, default=0.9)
    ap.add_argument("--stage1-conf-sweep", default=None,
                    help="逗號分隔的門檻清單，逐一評估 stage1")
    ap.add_argument("--stage2-model", default=None,
                    help="第二階段模型（空=不啟用兩段式）；填小模型路徑＝同顆模型零延遲")
    ap.add_argument("--stage2-conf", type=float, default=0.5)
    ap.add_argument("--verify-prefixes", default=None,
                    help="逗號分隔；第二階段接受的前綴詞（預設取 Config 值）")
    ap.add_argument("--verify-variants", default=None,
                    help="逗號分隔；第二階段接受的 hermes 變體詞（預設取 Config 值）")
    ap.add_argument("--verify-window-sec", type=float, default=1.5)
    ap.add_argument("--verify-interval-sec", type=float, default=0.25)
    ap.add_argument("--preroll-sec", type=float, default=1.0)
    ap.add_argument("--gate-words", default=None, help="逗號分隔；hey 閘門詞彙（預設 hey,hay）")
    ap.add_argument("--gate-conf", type=float, default=0.5)
    ap.add_argument("--no-gate-partial", action="store_true",
                    help="閘門只用定案結果（不用 partial）")
    ap.add_argument("--cascade", action="store_true",
                    help="評測串接式（hey 閘門→hermes 確認）並量喚醒延遲")
    ap.add_argument("--oww-model",
                    default="~/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx")
    ap.add_argument("--oww-threshold", type=float, default=0.6)
    ap.add_argument("--oww-confirmation-frames", type=int, default=3)
    ap.add_argument("--oww-vad-threshold", type=float, default=0.0)
    ap.add_argument("--oww-threshold-sweep", default=None,
                    help="逗號分隔的 openWakeWord 門檻清單，逐一評估")
    ap.add_argument("--quiet", action="store_true", help="只印矩陣，不印誤判來源")
    args = ap.parse_args()

    corpus = Path(args.corpus).expanduser()
    if not (corpus / "labels.jsonl").exists():
        sys.exit(f"找不到語料：{corpus}/labels.jsonl（先跑 build_wake_corpus.py）")

    engine = "cascade" if args.cascade else args.engine
    if engine == "cascade":
        cfg = Config()
        cfg.audio.blocksize = BLOCK
        if args.gate_words:
            cfg.wake.vosk_words = args.gate_words.split(",")
        cfg.wake.vosk_min_conf = args.gate_conf
        cfg.wake.gate_partial = not args.no_gate_partial
        cfg.wake.verify_window_sec = args.verify_window_sec
        cfg.wake.verify_interval_sec = args.verify_interval_sec
        cfg.wake.preroll_sec = args.preroll_sec
        cfg.wake.verify_model = args.stage2_model or ""
        cfg.wake.verify_min_conf = args.stage2_conf
        res = _run_cascade(corpus, cfg)
        _print(res, verbose=not args.quiet)
        return

    if engine == "oww":
        thresholds = [args.oww_threshold]
        if args.oww_threshold_sweep:
            thresholds = [float(value) for value in args.oww_threshold_sweep.split(",")]
        for threshold in thresholds:
            result = _run_oww(
                corpus,
                args.oww_model,
                threshold,
                args.oww_confirmation_frames,
                args.oww_vad_threshold,
            )
            _print(result, verbose=not args.quiet)
        return

    confs = [args.stage1_conf]
    if args.stage1_conf_sweep:
        confs = [float(x) for x in args.stage1_conf_sweep.split(",")]
    prefixes = args.verify_prefixes.split(",") if args.verify_prefixes else None
    variants = args.verify_variants.split(",") if args.verify_variants else None
    s1w = args.stage1_words.split(",") if args.stage1_words else None

    for c in confs:
        res = _run(corpus, args.stage1_model, c,
                   args.stage2_model, args.stage2_conf, args.verify_window_sec,
                   prefixes=prefixes, variants=variants, stage1_words=s1w)
        _print(res, verbose=not args.quiet)


if __name__ == "__main__":
    main()
