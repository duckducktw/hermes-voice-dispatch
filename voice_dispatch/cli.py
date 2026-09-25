"""命令列進入點：argparse + 記錄設定 + 分派模式。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import audio
from .config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice_dispatch",
        description="Hermes 語音派工守護程式（拍手兩下 + 喊 Hermes 喚醒）",
    )
    p.add_argument("--config", "-c", help="YAML 設定檔路徑（可省略，用內建預設值）")
    p.add_argument("--once", action="store_true", help="只跑一輪就結束")
    p.add_argument("--simulate", metavar="TEXT", help="跳過麥克風，直接以此文字走後段流程")
    p.add_argument("--dry-run", action="store_true",
                   help="只做本地流程、印出將發送內容，不打 Discord、不 spawn hermes")
    p.add_argument("--list-devices", action="store_true", help="列出可用的麥克風裝置後結束")
    p.add_argument("--check-audio", action="store_true",
                   help="健檢麥克風擷取路徑（死訊號/crest 判定）後結束")
    p.add_argument("--log-level", default="INFO",
                   help="log 等級（DEBUG/INFO/WARNING/ERROR），預設 INFO")
    p.add_argument("--log-file", help="覆寫 log 檔路徑")
    return p


def setup_logging(level: str, log_file: Optional[str]) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        try:
            Path(os.path.dirname(log_file) or ".").mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError as exc:
            print(f"[警告] 無法寫入 log 檔 {log_file}：{exc}", file=sys.stderr)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # --list-devices 不需要 config / token，最優先處理
    if args.list_devices:
        print(audio.format_device_list())
        return 0

    # dry-run 不碰網路，不需要 token；其餘模式（含 simulate 真跑）仍載入 token。
    # 注意：載入時即使找不到 token 也只是留 None，不會報錯（真的要用時才在 DiscordClient 檢查）。
    need_token = not args.dry_run and not args.check_audio
    try:
        cfg: Config = load_config(args.config, load_token=need_token)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[錯誤] 載入設定失敗：{exc}", file=sys.stderr)
        return 2

    log_file = args.log_file or cfg.resolved_log_file()
    setup_logging(args.log_level, log_file)

    if args.check_audio:
        m = audio.measure_level(cfg.audio, seconds=2.0)
        print(f"輸入裝置：{audio.describe_device(cfg.audio.device)}")
        if m:
            print(f"  rms={m['rms']:.5f}  peak={m['peak']:.5f}  "
                  f"crest={m['crest']:.2f}  samples={m['samples']}")
        print(f"判定：{audio.level_verdict(m)}")
        print("（crest < 2 = 死訊號，調門檻沒用，要換節點或修驅動；"
              "正常講話 crest 應 5~8）")
        ok = bool(m) and m["crest"] >= 2.0 and m["peak"] >= 0.01
        return 0 if ok else 1

    dispatcher = _make_dispatcher(cfg, dry_run=args.dry_run)

    if args.simulate is not None:
        return dispatcher.simulate(args.simulate)
    if args.once:
        return dispatcher.run_once()
    return dispatcher.run()


def _make_dispatcher(cfg: Config, *, dry_run: bool):
    # 延遲 import，避免 --list-devices 這種輕量路徑載入整條依賴
    from .daemon import VoiceDispatcher

    return VoiceDispatcher(cfg, dry_run=dry_run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
