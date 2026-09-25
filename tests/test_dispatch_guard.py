"""prompt_and_capture 的守門：STT 標記為「疑似幻覺」的結果不可以當成需求回傳。

2026-09-25 實際發生：STT 把一段雜音幻覺成「謝謝觀看,下次見。」，daemon 照樣
發了一張假任務卡到 #人工智障。這組測試把「絕不派工幻覺」鎖住。
"""
import numpy as np

from voice_dispatch import stt
from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher


def _prep(monkeypatch, text):
    d = VoiceDispatcher(Config(), dry_run=True)
    monkeypatch.setattr(d, "_record_utterance",
                        lambda stream: np.ones(16000, dtype=np.float32))
    monkeypatch.setattr(stt, "transcribe_samples", lambda *a, **k: text)
    return d


def test_hallucination_marked_text_rejected(monkeypatch):
    d = _prep(monkeypatch,
              "⚠️（疑似辨識幻覺，內容不可信，請重錄或改打字）原始輸出：謝謝觀看,下次見。")
    assert d.prompt_and_capture(None) is None


def test_normal_text_passes(monkeypatch):
    d = _prep(monkeypatch, "幫我把伺服器重啟")
    assert d.prompt_and_capture(None) == "幫我把伺服器重啟"


def test_empty_text_rejected(monkeypatch):
    d = _prep(monkeypatch, "   ")
    assert d.prompt_and_capture(None) is None
