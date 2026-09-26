#!/usr/bin/env python3
"""探針：驗證 Hermes 內建 api_server 能不能當語音派工入口。

用法：
    ~/.hermes/hermes-agent/venv/bin/python3 tools/probe_api_server.py "你好，自我介紹一句"

會做四件事並把「原始回應」印出來（不美化，方便看 schema）：
  1. GET  /health
  2. GET  /v1/models
  3. POST /v1/chat/completions  (stream=true, 帶 X-Hermes-Session-Id) → 逐行印 SSE
  4. POST /v1/runs → run_id，再 GET /v1/runs/{id}/events → 逐行印 SSE

key 從 ~/.hermes/.env 的 API_SERVER_KEY 讀；port 預設 8642。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("HERMES_API_BASE", "http://127.0.0.1:8642")


def _read_key() -> str:
    for env in ("API_SERVER_KEY",):
        v = os.environ.get(env)
        if v:
            return v.strip()
    p = Path("~/.hermes/.env").expanduser()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("API_SERVER_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("找不到 API_SERVER_KEY（環境變數或 ~/.hermes/.env）")


KEY = _read_key()


def req(method: str, path: str, body: dict | None = None, *, stream: bool = False, sid: str | None = None):
    url = BASE + path
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if sid:
        headers["X-Hermes-Session-Id"] = sid
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    return urllib.request.urlopen(r, timeout=120)


def show(title: str, method: str, path: str, body: dict | None = None, *, stream=False, sid=None):
    print(f"\n{'='*70}\n### {title}\n{method} {path}" + (f"  (sid={sid})" if sid else ""))
    try:
        resp = req(method, path, body, stream=stream, sid=sid)
    except urllib.error.HTTPError as e:
        print(f"!! HTTP {e.code}: {e.read().decode('utf-8','replace')[:500]}")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"!! {type(e).__name__}: {e}")
        return None
    print(f"HTTP {resp.status}  content-type={resp.headers.get('content-type')}")
    if stream:
        n = 0
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line:
                print("  |", line[:300])
                n += 1
            if n > 40:
                print("  | ...(省略)")
                break
        return None
    raw = resp.read().decode("utf-8", "replace")
    print(raw[:1200])
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "用一句話自我介紹"
    print(f"BASE={BASE}")
    show("health", "GET", "/health")
    show("models", "GET", "/v1/models")

    sid = "voice-probe-001"
    show(
        "chat/completions (stream, 同一 session 連續兩輪)",
        "POST", "/v1/chat/completions",
        {"model": "hermes-agent", "stream": True,
         "messages": [{"role": "user", "content": prompt}]},
        stream=True, sid=sid,
    )
    show(
        "chat/completions 第二輪（驗 session 連續）",
        "POST", "/v1/chat/completions",
        {"model": "hermes-agent", "stream": True,
         "messages": [{"role": "user", "content": "剛剛我問你什麼？一句話。"}]},
        stream=True, sid=sid,
    )

    run = show("runs (start)", "POST", "/v1/runs",
               {"input": prompt, "session_id": sid})
    if isinstance(run, dict) and run.get("run_id"):
        show("runs/{id}/events (SSE)", "GET", f"/v1/runs/{run['run_id']}/events", stream=True)


if __name__ == "__main__":
    main()
