"""探針：走一次 tts.speak 的「分段合成 → 合併」管線（**不播放**），驗證完整不截斷。

用法：PYTHONPATH=. ~/.hermes/hermes-agent/venv/bin/python3 tools/probe_speak_pipeline.py [字數]
"""
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_dispatch.config import load_config          # noqa: E402
from voice_dispatch import tts                          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("probe")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
sentence = "這是一段用來驗證語音回報不會被截斷的測試句子，長度固定。"
text = (sentence * (n // len(sentence) + 1))[:n]

cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"))
print(f"speak_result_max_chars={cfg.tts.speak_result_max_chars} "
      f"speak_chunk_chars={cfg.tts.speak_chunk_chars} speak_cps={cfg.tts.speak_cps}")

chunks = tts.split_speech_chunks(text, cfg.tts.speak_chunk_chars)
print(f"目標字數={len(text)} → 切成 {len(chunks)} 段（各段長度 {[len(c) for c in chunks]}）")

base = os.path.join(tempfile.gettempdir(), "vd_probe")
out = base + ".mp3"
engine = (cfg.tts.engine or "edge").lower()
parts = []
for i, ch in enumerate(chunks):
    p = f"{base}.part{i}.mp3"
    tts._synth_unit(ch, p, cfg, engine, logger=log)
    tts._apply_pace(p, ch, cfg, logger=log)
    parts.append(p)
tts._concat_audio(parts, out, logger=log)
for p in parts:
    try:
        os.remove(p)
    except OSError:
        pass

dur = tts._duration(out)
chars = len([c for c in text if not c.isspace()])
print(f"結果：秒數={dur:.2f} 字/秒={chars / dur if dur else 0:.2f} 檔={out}")
expect = chars / max(cfg.tts.speak_cps, 0.1)
print(f"預期秒數≈{expect:.1f} → {'完整' if dur >= expect * 0.7 else '疑似被截斷'}")
