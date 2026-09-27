"""探針：長文中譯走 tts.synth_gemini_to_file，看會不會被截斷。

用法：PYTHONPATH=. ~/.hermes/hermes-agent/venv/bin/python3 tools/probe_long_tts.py [字數]
判定：印出實際音檔秒數與字/秒；若 字/秒 掉到 <3 或音檔明顯短於預期 → 模型截斷。
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_dispatch import config as cfgmod  # noqa: E402
from voice_dispatch import tts  # noqa: E402

log = logging.getLogger("probe")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

BASE = (
    "查完了，結論是這樣：第一個問題在於標籤會 fail-open，標籤一被動或拔掉，"
    "系統就當沒這回事直接放行，所以要先把它改成 fail-closed。"
    "第二個問題是跨平台撞名，你的 composite group 在解析的時候把路由卡死，"
    "我已經把重複的那一筆清掉，重新載入之後就通了。"
    "第三個是伺服器的 TPS 掉到十二，主要來源是實體 AI 太密集，"
    "建議把附近實體的 tick 間隔調高，並且把掛機點集中到一個區塊。"
    "第四，備份排程目前是凌晨三點跑 restic，但最近一次有兩百多 MB 的增量，"
    "硬碟還剩四成空間，暫時不用處理。"
)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cfg = cfgmod.load_config("config.yaml")
    text = (BASE * ((n // len(BASE)) + 1))[:n]
    print(f"目標字數={len(text)}")
    out = "/tmp/probe_long_tts.mp3"
    if os.path.exists(out):
        os.remove(out)
    try:
        tts.synth_gemini_to_file(text, out, cfg, logger=log)
    except Exception as exc:  # noqa: BLE001
        print(f"合成失敗：{exc}")
        return 1
    dur = tts._duration(out)
    cps = tts._cps(out, len([c for c in text if not c.isspace()]))
    print(f"結果：bytes={os.path.getsize(out)} 秒數={dur:.2f} 字/秒={cps:.2f}")
    print(f"判定：{'疑似截斷（字/秒 < 3.0）' if 0 < cps < 3.0 else '看起來完整'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
