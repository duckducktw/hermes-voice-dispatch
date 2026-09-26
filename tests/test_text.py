"""純文字邏輯測試：喚醒詞、同意/不同意、thread 名截斷。"""

from voice_dispatch.config import Config
from voice_dispatch.text import (
    classify_confirmation,
    contains_any,
    is_wake_transcript,
    make_thread_name,
    normalize,
    spoken_summary,
    truncate_thread_name,
)


def test_normalize():
    assert normalize("Hello, World!") == "helloworld"
    assert normalize("赫米斯。") == "赫米斯"
    assert normalize("") == ""


def test_wake_keyword_hits():
    kw = Config().wake.keywords
    assert is_wake_transcript("嘿 Hermes 你好", kw) is True
    assert is_wake_transcript("HERMES!", kw) is True
    assert is_wake_transcript("赫米斯", kw) is True
    assert is_wake_transcript("哈米斯，在嗎", kw) is True


def test_wake_keyword_misses():
    kw = Config().wake.keywords
    assert is_wake_transcript("今天天氣真好", kw) is False
    assert is_wake_transcript("", kw) is False


def test_confirmation_agree():
    c = Config().confirm
    for text in ["對", "是的", "好啊", "沒錯", "可以", "OK", "嗯"]:
        assert classify_confirmation(text, c.agree_words, c.disagree_words) == "agree", text


def test_confirmation_disagree_priority():
    c = Config().confirm
    # 否定詞優先：「不對」不能被判成 agree
    assert classify_confirmation("不對", c.agree_words, c.disagree_words) == "disagree"
    for text in ["不要", "錯了", "重講", "再說一次", "重來"]:
        assert classify_confirmation(text, c.agree_words, c.disagree_words) == "disagree", text


def test_confirmation_unknown():
    c = Config().confirm
    assert classify_confirmation("隨便啦", c.agree_words, c.disagree_words) == "unknown"
    assert classify_confirmation("", c.agree_words, c.disagree_words) == "unknown"


def test_truncate_thread_name():
    assert truncate_thread_name("abc", 100) == "abc"
    long = "字" * 150
    assert len(truncate_thread_name(long, 100)) == 100
    assert truncate_thread_name("abc", 0) == ""


def test_make_thread_name_within_limit():
    long_req = "幫我把伺服器重啟" * 30
    name = make_thread_name(long_req, "09-25 14:30", short_len=20, limit=100)
    assert len(name) <= 100
    assert "09-25 14:30" in name


def test_make_thread_name_short():
    name = make_thread_name("重啟伺服器", "09-25 14:30")
    assert name.startswith("🎙️")
    assert "重啟伺服器" in name


# ── spoken_summary（派工完成後語音回報用）────────────────────────────────
def test_spoken_summary_strips_markdown():
    t = "**完成**：\n- 已把 `bright.py` 跑完\n- rc=0"
    assert spoken_summary(t) == "完成： 已把 bright.py 跑完 rc=0"


def test_spoken_summary_truncates_at_sentence():
    t = "第一句話講完了。第二句話很長" + "啊" * 200
    out = spoken_summary(t, max_chars=12)
    assert out.endswith("。")
    assert len(out) <= 20


def test_spoken_summary_short_passthrough():
    assert spoken_summary("好了", 160) == "好了"
    assert spoken_summary("", 160) == ""
