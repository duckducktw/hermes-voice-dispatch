"""pick_report：(c) 方案「重新武裝」時該唸哪一則的挑選邏輯（純函式）。"""
from __future__ import annotations

from voice_dispatch.daemon import pick_report


def test_picks_latest_stop_with_content():
    """多回合：要挑『最新』那則最終回覆（不是第一回合那則）。"""
    rows = [
        (1, "assistant", "先講關鍵：…", "tool_calls"),
        (2, "assistant", "看完了。這支 log 是…", "stop"),
        (3, "session_meta", "", ""),
        (4, "assistant", "**是有關的 —— 那支 log 就是我裝的**", "stop"),
        (5, "session_meta", "", ""),
    ]
    assert pick_report(rows, 0) == (4, "**是有關的 —— 那支 log 就是我裝的**")


def test_skips_already_spoken():
    """已唸過的（id <= last_spoken_id）不能再唸 → 取下一則。"""
    rows = [
        (2, "assistant", "第一輪結果", "stop"),
        (4, "assistant", "第二輪結果", "stop"),
    ]
    assert pick_report(rows, 2) == (4, "第二輪結果")


def test_none_when_nothing_new():
    assert pick_report([(2, "assistant", "第一輪結果", "stop")], 2) is None
    assert pick_report([], 0) is None


def test_skips_empty_content_and_non_stop():
    rows = [
        (1, "assistant", "", "stop"),             # 空內容不唸
        (2, "assistant", "   ", "stop"),          # 只有空白也不唸
        (3, "assistant", "中間敘述", "tool_calls"),  # 非 stop 不唸
        (4, "tool", "tool output", ""),           # 非 assistant 不唸
        (5, "session_meta", "", ""),
    ]
    assert pick_report(rows, 0) is None
