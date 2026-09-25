"""kws.py：喚醒詞偵測與 Silero VAD 的基本行為（不需麥克風）。"""
import numpy as np

from voice_dispatch import kws


def test_silence_does_not_trigger_wake():
    """純靜音不該命中——常開聽最重要的性質就是不要亂觸發。"""
    sp = kws.WakeWordSpotter(["hey_jarvis"], threshold=0.5)
    for _ in range(20):        # 20 × 80ms = 1.6 秒
        assert sp.feed(np.zeros(1280, dtype=np.float32)) is None


def test_chunker_only_emits_full_chunks():
    """碎片不推論：湊滿一個 80ms 區塊才送模型。"""
    c = kws._Chunker()
    assert c.push(np.zeros(700, dtype=np.float32)) == []
    assert len(c.push(np.zeros(700, dtype=np.float32))) == 1


def test_silero_none_until_full_chunk():
    """Silero 只吃剛好 1280 samples；不足時回 None（上層退回 RMS 門檻）。"""
    v = kws.SileroVad(0.5)
    assert v.feed(np.zeros(512, dtype=np.float32)) is None
    assert v.feed(np.zeros(1024, dtype=np.float32)) in (True, False)
