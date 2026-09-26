"""Discord REST：發訊息、建 public thread、發到 thread。

用 stdlib urllib（不引入 requests）。錯誤（401/403/429…）給明確訊息。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .config import Config


class DiscordError(RuntimeError):
    """Discord API 呼叫失敗。"""


class DiscordClient:
    def __init__(self, cfg: Config, logger=None):
        self.cfg = cfg
        self.logger = logger
        token = cfg.discord_token
        if not token:
            raise DiscordError(
                f"缺少 Discord bot token（環境變數 {cfg.discord.token_env} "
                f"或 {cfg.discord.env_file}）"
            )
        self.token = token
        self.base = cfg.discord.api_base.rstrip("/")

    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        data = None
        headers = {
            "Authorization": f"Bot {self.token}",
            "User-Agent": "hermes-voice-dispatch (https://example.invalid, 0.1.0)",
            "Accept": "application/json",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.discord.timeout_sec) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace") if exc.fp else ""
            self._raise_http(exc.code, body, method, path)
        except urllib.error.URLError as exc:
            raise DiscordError(f"連線 Discord 失敗（{method} {path}）：{exc.reason}") from exc

        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def _raise_http(self, code: int, body: str, method: str, path: str) -> None:
        snippet = body[:500]
        if code == 401:
            raise DiscordError(f"Discord 401 未授權（token 無效？）：{snippet}")
        if code == 403:
            raise DiscordError(
                f"Discord 403 權限不足（bot 缺少發言/建討論串權限？）：{snippet}"
            )
        if code == 404:
            raise DiscordError(f"Discord 404 找不到目標（頻道 id 錯誤？）：{snippet}")
        if code == 429:
            retry_after = ""
            try:
                retry_after = str(json.loads(body).get("retry_after", ""))
            except Exception:  # noqa: BLE001
                pass
            raise DiscordError(
                f"Discord 429 觸發速率限制，retry_after={retry_after}s：{snippet}"
            )
        raise DiscordError(f"Discord API 錯誤 {code}（{method} {path}）：{snippet}")

    # ------------------------------------------------------------------
    def post_message(self, channel_id: str, content: str) -> Dict[str, Any]:
        """發訊息到頻道，回傳訊息物件（含 id）。"""
        return self._request(
            "POST", f"/channels/{channel_id}/messages", {"content": content}
        )

    def create_thread_from_message(
        self, channel_id: str, message_id: str, name: str,
        auto_archive_duration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """由某則訊息建立 public thread，回傳 thread 物件（含 id）。"""
        payload = {
            "name": name,
            "auto_archive_duration": (
                auto_archive_duration
                if auto_archive_duration is not None
                else self.cfg.discord.auto_archive_duration
            ),
        }
        return self._request(
            "POST",
            f"/channels/{channel_id}/messages/{message_id}/threads",
            payload,
        )

    def post_thread_message(self, thread_id: str, content: str) -> Dict[str, Any]:
        """發訊息到 thread（thread 本身也是一個 channel）。"""
        return self.post_message(thread_id, content)

    def get_messages(self, channel_id: str, limit: int = 10) -> Any:
        """讀頻道/討論串最近訊息（回傳 list）。用於判斷 agent 有沒有回報過。"""
        return self._request("GET", f"/channels/{channel_id}/messages?limit={int(limit)}")
