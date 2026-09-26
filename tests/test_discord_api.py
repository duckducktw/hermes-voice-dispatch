"""discord_api 的小工具測試（不打真網路）。"""
from __future__ import annotations

from voice_dispatch.discord_api import user_avatar_url


def test_user_avatar_url_png():
    u = {"id": "123", "avatar": "abc"}
    assert user_avatar_url(u) == "https://cdn.discordapp.com/avatars/123/abc.png?size=128"


def test_user_avatar_url_animated_gif():
    u = {"id": "123", "avatar": "a_xyz"}
    assert user_avatar_url(u).endswith("/123/a_xyz.gif?size=128")


def test_user_avatar_url_empty_when_missing():
    assert user_avatar_url({}) == ""
    assert user_avatar_url({"id": "123"}) == ""
    assert user_avatar_url({"id": "", "avatar": "abc"}) == ""
