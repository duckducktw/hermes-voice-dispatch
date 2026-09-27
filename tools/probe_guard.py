"""探針：驗證「模型截斷守門」（tts._synth_unit）在單次超長文時會自動切半救回。

用法：PYTHONPATH=. ~/.hermes/hermes-agent/venv/bin/python3 tools/probe_guard.py [字數]
"""
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_dispatch.config import load_config          # noqa: E402
from voice_dispatch import tts                          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("guard")

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
sentence = "這是一段用來驗證語音回報不會被截斷的測試句子，長度固定。"
text = (sentence * (n // len(sentence) + 1))[:n]

cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"))
out = os.path.join(tempfile.gettempdir(), "vd_guard.mp3")
print(f"（故意關掉分段，強迫單次合成，看守門會不會自己救）")
tts._synth_unit(text, out, cfg, (cfg.tts.engine or "edge").lower(), logger=log)
dur = tts._duration(out)
chars = len([c for c in text if not c.isspace()])
cps = chars / dur if dur else 0
print(f"結果：{chars} 字 → {dur:.2f}s（{cps:.2f} 字/秒）")
print("判定：", "完整（守門救回，接近真實語速）" if cps < cfg.tts.speak_cps * 1.6
      else "仍偏快 → 守門沒生效")
