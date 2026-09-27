"""長文分段合成（tts.split_speech_chunks）的純邏輯測試。

背景（2026-09-27 使用者）：「它現在都會截斷訊息，完整的訊息不會唸出來」
→ 語音回報不再截斷，長文交由 tts.speak 分句切段合成再接起來。
"""

from voice_dispatch.tts import split_speech_chunks


def test_short_text_not_split():
    assert split_speech_chunks("你好。", 100) == ["你好。"]


def test_limit_zero_means_no_split():
    t = "啊" * 500
    assert split_speech_chunks(t, 0) == [t]


def test_empty_text():
    assert split_speech_chunks("", 100) == []
    assert split_speech_chunks("   ", 100) == []


def test_split_at_sentence_keeps_punctuation_and_order():
    t = "第一句話。" * 60          # 每句 5 字 → 每段 4 句（20 字）
    chunks = split_speech_chunks(t, 20)
    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks) == t            # 內容不遺漏、順序不變
    assert len(chunks) > 1


def test_single_overlong_sentence_is_hard_split():
    t = "啊" * 55
    chunks = split_speech_chunks(t, 20)
    assert [len(c) for c in chunks] == [20, 20, 15]
    assert "".join(chunks) == t


def test_mixed_short_and_long_sentences():
    t = "短句。" + "長" * 50 + "收尾。"
    chunks = split_speech_chunks(t, 20)
    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks) == t
