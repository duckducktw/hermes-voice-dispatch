"""派工：組 dispatch_prompt，背景 spawn `hermes -z "<prompt>"`。

背景程序須 detached、不阻塞守護迴圈，stdout/stderr 導到 log 檔。
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from .config import Config


def build_dispatch_prompt(
    transcript: str, channel_id: str, thread_id: str, cfg: Config
) -> str:
    """套用模板組出給 hermes agent 的派工 prompt。"""
    return cfg.dispatch.prompt_template.format(
        transcript=transcript,
        channel_id=channel_id,
        thread_id=thread_id,
    )


def build_hermes_command(prompt: str, cfg: Config) -> List[str]:
    """組出背景執行的 argv。"""
    return [cfg.dispatch.hermes_bin, "-z", prompt]


def _log_path(cfg: Config, thread_id: str) -> str:
    log_dir = cfg.resolved_dispatch_log_dir()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(log_dir, f"dispatch-{stamp}-{thread_id}.log")


def spawn_hermes(
    prompt: str, cfg: Config, thread_id: str, logger=None
) -> int:
    """背景（detached）啟動 hermes；回傳 pid。stdout/stderr 導到 log 檔。"""
    argv = build_hermes_command(prompt, cfg)
    log_file = _log_path(cfg, thread_id)
    # 用檔案代替 pipe，避免子程序寫滿 pipe 而卡住
    fh = open(log_file, "ab", buffering=0)
    try:
        header = f"# dispatch spawned {datetime.now().isoformat()}\n# argv={argv!r}\n"
        fh.write(header.encode("utf-8"))
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=fh,
            start_new_session=True,  # detach：不隨 daemon 終止而被送 SIGINT
        )
    finally:
        # 子程序已持有 fd，父程序可關閉自己的 handle
        fh.close()
    if logger:
        logger.info("已背景派工 hermes（pid=%s），log：%s", proc.pid, log_file)
    return proc.pid
