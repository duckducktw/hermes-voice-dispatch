"""派工：組 dispatch_prompt，背景 spawn `hermes -z "<prompt>"`。

背景程序須 detached、不阻塞守護迴圈，stdout/stderr 導到 log 檔。
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    prompt: str, cfg: Config, thread_id: str, logger=None, on_exit=None,
    heartbeat=None, heartbeat_sec: float = 0.0,
) -> int:
    """背景（detached）啟動 hermes；回傳 pid。stdout/stderr 導到 log 檔。

    - `on_exit(proc, log_file)`：可選回呼，程序結束後在**背景執行緒**呼叫
      （用來保底把最終輸出貼回 Discord，見 daemon 的用法）。
    - `heartbeat(秒數)` + `heartbeat_sec>0`：程序還在跑時每 N 秒呼叫一次
      （用來在串內發「還在處理中」心跳，讓使用者看得到東西有在動）。
    """
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
    if on_exit is not None or (heartbeat is not None and heartbeat_sec > 0):
        th = threading.Thread(
            target=_watch,
            args=(proc, log_file, on_exit, heartbeat, float(heartbeat_sec)),
            daemon=True,
        )
        th.start()
    return proc.pid


def _watch(proc, log_file: str, on_exit, heartbeat, heartbeat_sec: float) -> None:
    """背景看護：程序還在跑時每 `heartbeat_sec` 秒回呼一次 `heartbeat(秒數)`；
    結束後呼叫 `on_exit(proc, log_file)`。任何例外都吞掉（不干擾主流程）。"""
    start = time.monotonic()
    while True:
        try:
            if heartbeat is not None and heartbeat_sec > 0:
                proc.wait(timeout=heartbeat_sec)
                break
            proc.wait()
            break
        except subprocess.TimeoutExpired:
            try:
                heartbeat(int(time.monotonic() - start))
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            break
    if on_exit is not None:
        try:
            on_exit(proc, log_file)
        except Exception:  # noqa: BLE001
            pass


def extract_stdout(log_file: str) -> str:
    """從 dispatch log 取出 agent 的 stdout（濾掉我們自己寫的 `# ...` 表頭行）。"""
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return ""
    lines = [ln for ln in raw.splitlines()
             if not ln.startswith("#") and not ln.startswith("dispatch spawned")
             and not ln.startswith("argv=")]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Relay 派工：把語音需求當成「一則 @Hermes 的使用者訊息」**直接發在母頻道**
# ---------------------------------------------------------------------------
def build_relay_content(transcript: str, cfg: Config) -> str:
    """組出 relay 訊息內容。

    有 `relay.mention_id` 時＝`<@gateway_bot> 需求原文`；**留空＝純需求原文（完全不帶 @）**。

    為什麼要 @：gateway 的 `DISCORD_ALLOW_BOTS=mentions` 對 **bot 作者**另有一道閘
    （`plugins/platforms/discord/adapter.py:1347`
    `if allow_bots == "mentions" and not self._self_is_explicitly_mentioned(message): return False`），
    只收「@提及 Hermes 的 bot 訊息」。`free_response_channels` 只讓**人類**免 @。

    2026-09-26 使用者要求「能改成不 @嗎」→ 做法是 `.env` 把 `DISCORD_ALLOW_BOTS` 改成
    非 `none`/`mentions` 的值（本機用 `all`；改了那道閘就不作用），然後把
    `relay.mention_id` 留空。⚠️ 兩者要配套：只留空 mention 而 .env 未生效時，gateway
    不會接手 → daemon 會等不到討論串而**回退 spawn**（仍會完成，但多一張卡）。
    """
    mid = str(getattr(cfg.relay, "mention_id", "") or "").strip()
    return f"<@{mid}> {transcript}" if mid else transcript


def build_relay_payload(
    transcript: str, cfg: Config, identity: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """組 webhook POST 的 body。

    `identity`＝要模仿的發話身分 `{"username": …, "avatar_url": …}`。使用者要求
    「模仿 bot 在伺服器的外觀」，所以 daemon 會先用 `GET /users/@me` 取 Hermes bot
    自己的名字＋頭像傳進來；沒給就退回設定裡的 `relay.username`。
    """
    payload: Dict[str, Any] = {"content": build_relay_content(transcript, cfg)}
    ident = identity or {}
    username = str(ident.get("username") or cfg.relay.username or "").strip()
    if username:
        payload["username"] = username
    avatar = str(ident.get("avatar_url") or "").strip()
    if avatar:
        payload["avatar_url"] = avatar
    return payload


def post_relay(
    transcript: str, cfg: Config, logger=None,
    identity: Optional[Dict[str, str]] = None,
) -> dict:
    """用 Discord webhook 把需求直接發在**母頻道**（不帶 thread_id＝發在頻道本身）。

    gateway 會（因 `force_thread_channels`）自己把這則變成討論串的開頭並在串內回覆。
    成功回傳 Discord 的訊息 JSON（wait=true；含 id，可能已帶 `thread`）。失敗會
    raise，讓呼叫端回退 spawn。
    """
    import json
    import urllib.error
    import urllib.request

    url = cfg.relay_url or ""
    if not url:
        raise RuntimeError("relay url 為空（.env 沒有 DISCORD_RELAY_WEBHOOK_URL）")

    sep = "&" if "?" in url else "?"
    target = f"{url}{sep}wait=true"
    payload = build_relay_payload(transcript, cfg, identity)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        target, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            # 必須自帶 User-Agent：Discord 前面的 Cloudflare 會擋 urllib 預設 UA
            # （實測回應 403 body「error code: 1010」）。既有 DiscordClient 同樣做法。
            "User-Agent": "hermes-voice-dispatch (https://example.invalid, 0.1.0)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.relay.timeout_sec)) as resp:
            raw = resp.read().decode("utf-8", "replace")
            data = json.loads(raw) if raw else {}
            if logger:
                logger.info("relay 訊息已送出（HTTP %s，msg=%s）",
                            resp.status, data.get("id"))
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"relay HTTP {exc.code}：{detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"relay 送出失敗：{exc}") from exc

