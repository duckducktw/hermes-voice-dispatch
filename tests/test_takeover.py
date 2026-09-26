"""human_took_over：偵測「任務被真人接管」（打字／DC 語音）→ 不要 TTS 回報。"""
from __future__ import annotations

from voice_dispatch.daemon import human_took_over

BOT = {"id": "1520796555580543138", "bot": True, "username": "hermes agent"}
HUMAN = {"id": "1058750638760149033", "bot": False, "username": "鴨咧鴨咧"}


def _m(mtype=0, author=None, webhook_id=None, content="x"):
    m = {"type": mtype, "author": author if author is not None else BOT, "content": content}
    if webhook_id:
        m["webhook_id"] = webhook_id
    return m


def test_no_human_means_not_taken_over():
    """只有 bot、webhook（模仿 bot 發的 relay）、系統訊息 → 沒被接管。"""
    msgs = [
        _m(21, BOT, content=""),                      # 串首
        _m(1, BOT, content=""),                       # type1 已加入成員
        _m(0, BOT, webhook_id="1553326360368513056"),  # 我模仿 bot 發的 relay
        _m(0, BOT, content="agent 回覆"),
        _m(19, BOT, content="agent 回覆"),
    ]
    assert human_took_over(msgs) is False


def test_typed_message_means_taken_over():
    """串內出現真人打字 → 被接管（2026-09-26 使用者要求）。"""
    msgs = [_m(0, BOT, webhook_id="w"), _m(0, HUMAN, content="是不是合附魔消失有關")]
    assert human_took_over(msgs) is True


def test_voice_message_also_detected():
    """DC 語音（真人發的 type 0，內文常為空、帶附件）也算接管。"""
    assert human_took_over([_m(0, HUMAN, content="")]) is True


def test_human_reply_detected():
    assert human_took_over([_m(19, HUMAN)]) is True


def test_robust_to_garbage():
    assert human_took_over(None) is False
    assert human_took_over([]) is False
    assert human_took_over([None, "x", 1]) is False
    # 沒有 author.id 的殘缺物件不算
    assert human_took_over([_m(0, {"bot": False})]) is False
