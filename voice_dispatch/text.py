"""純文字邏輯：喚醒詞比對、同意/不同意判定、thread 名截斷。

全部是純函式，方便單元測試，不碰音訊/網路。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# 常見標點（中英），比對前先去除
_PUNCT_RE = re.compile(r"[\s。，、！？!?.,；;：:「」『』（）()\[\]【】…~～\-—_\"'`]+")


def normalize(text: str) -> str:
    """正規化：NFKC、轉小寫、去標點與空白。用於關鍵字比對。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    return text


def contains_any(text: str, words: Iterable[str]) -> bool:
    """正規化後，text 是否包含 words 中任一者。"""
    norm = normalize(text)
    if not norm:
        return False
    return any(normalize(w) and normalize(w) in norm for w in words)


def is_wake_transcript(transcript: str, keywords: Iterable[str]) -> bool:
    """轉錄文字是否命中喚醒詞。"""
    return contains_any(transcript, keywords)


def _longest_match(norm: str, words: Iterable[str]) -> int:
    """回傳 words 中「出現在 norm 內」且最長者的長度；沒有則 0。"""
    best = 0
    for w in words:
        nw = normalize(w)
        if nw and nw in norm and len(nw) > best:
            best = len(nw)
    return best


def classify_confirmation(
    transcript: str,
    agree_words: Iterable[str],
    disagree_words: Iterable[str],
) -> str:
    """回述確認的判定結果。

    回傳 "agree" / "disagree" / "unknown"。

    以「最長命中詞」決勝負，解決子字串衝突：
    - 「沒錯」（agree，長度 2）勝過「錯」（disagree，長度 1）→ agree
    - 「不對」中 agree「對」與 disagree「不」同為長度 1 → 平手時否定詞優先 → disagree
    """
    norm = normalize(transcript)
    if not norm:
        return "unknown"

    agree_len = _longest_match(norm, agree_words)
    disagree_len = _longest_match(norm, disagree_words)

    if agree_len == 0 and disagree_len == 0:
        return "unknown"
    if agree_len > disagree_len:
        return "agree"
    # disagree_len >= agree_len（含平手）→ 否定詞優先
    return "disagree"


def truncate_thread_name(name: str, limit: int = 100) -> str:
    """Discord thread 名有長度上限，超過就截斷。"""
    if limit <= 0:
        return ""
    if len(name) <= limit:
        return name
    return name[:limit]


def make_thread_name(
    transcript: str,
    date_str: str,
    template: str = "🎙️ {short} {date}",
    short_len: int = 20,
    limit: int = 100,
) -> str:
    """組出 thread 名並確保不超過上限。

    transcript 取前 short_len 字當摘要；最終長度受 limit 限制。
    """
    short = (transcript or "").strip().replace("\n", " ")
    if len(short) > short_len:
        short = short[:short_len]
    name = template.format(short=short, date=date_str)
    return truncate_thread_name(name, limit)


def spoken_summary(text: str, max_chars: int = 160) -> str:
    """把 Markdown 報告壓成一句「適合念出來」的短摘要（派工完成後用語音回報）。

    - 去掉 Markdown 裝飾（**粗體**、`code`、# 標題、- 項目符號）
    - 空白正規化
    - 超過 max_chars 就盡量切在句尾（否則補「…」）
    """
    t = re.sub(r"[*_`#>]+", "", text or "")
    t = re.sub(r"^\s*[-•・]\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    for sep in ("。", "！", "？", "；", ".", "!", "?", ";"):
        i = cut.rfind(sep)
        if i >= max_chars // 2:
            return cut[: i + 1]
    return cut.rstrip() + "…"
