"""relay 派工：內容組法與送出（不打真網路）。"""
from __future__ import annotations

import json
import urllib.error

import pytest

from voice_dispatch import dispatch
from voice_dispatch.config import Config


def _cfg():
    c = Config()
    c.relay.mention_id = "1234567890"
    c.relay_url = "https://discord.com/api/v10/webhooks/1/tok"
    return c


def test_build_relay_content_mentions_gateway_bot():
    """內容必須 @ gateway bot——ALLOW_BOTS=mentions 只收有 @提及 的 bot 訊息。"""
    out = dispatch.build_relay_content("幫我重啟伺服器", _cfg())
    assert out == "<@1234567890> 幫我重啟伺服器"


def test_post_relay_requires_url():
    c = _cfg()
    c.relay_url = ""
    with pytest.raises(RuntimeError, match="relay url"):
        dispatch.post_relay("hi", "999", c)


def test_post_relay_posts_into_thread(monkeypatch):
    """要帶 thread_id 且 body 有 content/username。"""
    seen = {}

    class _Resp:
        status = 200

        def read(self):
            return b'{"id": "42"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["ctype"] = req.headers.get("Content-type")
        seen["ua"] = req.headers.get("User-agent")
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    out = dispatch.post_relay("需求原文", "777", _cfg())
    assert out == {"id": "42"}
    assert "thread_id=777" in seen["url"]
    assert seen["body"]["content"] == "<@1234567890> 需求原文"
    assert seen["body"]["username"]
    assert seen["ctype"] == "application/json"
    # 一定要帶 User-Agent：Discord 前面 Cloudflare 會擋 urllib 預設 UA（403 error 1010）
    assert seen["ua"] and "urllib" not in seen["ua"].lower()


def test_post_relay_raises_on_http_error(monkeypatch):
    def _boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with pytest.raises(RuntimeError, match="relay HTTP 401"):
        dispatch.post_relay("x", "1", _cfg())
