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


def test_build_relay_content_is_mention_plus_requirement():
    """內容＝`<@gateway_bot> 需求原文`。

    必須 @ gateway bot——`DISCORD_ALLOW_BOTS=mentions` 只收有 @提及 的 bot 訊息。
    需求原文就寫在這一則裡（2026-09-26 定案：webhook 直接發母頻道，gateway 的
    `force_thread_channels` 會自己替它開討論串 → 全場只有這一則）。
    """
    out = dispatch.build_relay_content("幫我重啟伺服器", _cfg())
    assert out == "<@1234567890> 幫我重啟伺服器"


def test_build_relay_content_without_mention_when_id_empty():
    """`mention_id` 留空＝純需求原文（不帶 @）。前提見模組 docstring。"""
    c = _cfg()
    c.relay.mention_id = ""
    assert dispatch.build_relay_content("幫我重啟伺服器", c) == "幫我重啟伺服器"


def test_build_relay_payload_mimics_bot_identity():
    """模仿 bot 外觀：identity 的名字／頭像要蓋過設定值。"""
    payload = dispatch.build_relay_payload(
        "hi", _cfg(),
        identity={"username": "hermes agent", "avatar_url": "https://cdn/x.png"},
    )
    assert payload["username"] == "hermes agent"
    assert payload["avatar_url"] == "https://cdn/x.png"


def test_build_relay_payload_falls_back_to_configured_username():
    c = _cfg()
    c.relay.username = "🎙️ 語音輸入"
    payload = dispatch.build_relay_payload("hi", c)
    assert payload["username"] == "🎙️ 語音輸入"
    assert "avatar_url" not in payload


def test_post_relay_requires_url():
    c = _cfg()
    c.relay_url = ""
    with pytest.raises(RuntimeError, match="relay url"):
        dispatch.post_relay("hi", c)


def test_post_relay_posts_to_channel_not_thread(monkeypatch):
    """要發在**母頻道**（不帶 thread_id），回傳含 id 的訊息物件。"""
    seen = {}

    class _Resp:
        status = 200

        def read(self):
            return b'{"id": "42", "thread": {"id": "42"}}'

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
    out = dispatch.post_relay("需求原文", _cfg(), identity={"username": "hermes agent"})
    assert out["id"] == "42"
    assert "thread_id=" not in seen["url"], "要發在母頻道，不是發進討論串"
    assert "wait=true" in seen["url"]
    assert seen["body"]["content"] == "<@1234567890> 需求原文"
    assert seen["body"]["username"] == "hermes agent"
    assert seen["ctype"] == "application/json"
    # 一定要帶 User-Agent：Discord 前面 Cloudflare 會擋 urllib 預設 UA（403 error 1010）
    assert seen["ua"] and "urllib" not in seen["ua"].lower()


def test_post_relay_raises_on_http_error(monkeypatch):
    def _boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with pytest.raises(RuntimeError, match="relay HTTP 401"):
        dispatch.post_relay("x", _cfg())
