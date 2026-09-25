"""Daemon 的「只聽到否定詞才重錄」判定（fail-open 確認輪）測試。

2026-09-25：確認輪不再要求使用者說「對」，只有整句就是否定／重來詞才重錄。
"""
import pytest

from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher


@pytest.fixture()
def d():
    return VoiceDispatcher(Config(), dry_run=True)


@pytest.mark.parametrize("text", ["不對", "重來", "再說", "錯", "不要", "不對不對"])
def test_retry_only(d, text):
    assert d._is_retry_only(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "幫我把伺服器重啟",
        "我覺得這不行，換一個做法",   # 含「不」但不是單純否定 → 不可誤判成重錄
        "不要重啟，改成先備份",       # 有實際需求內容 → 不可誤判
        "",
    ],
)
def test_not_retry_only(d, text):
    assert d._is_retry_only(text) is False
