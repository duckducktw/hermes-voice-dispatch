"""守門測試：模型把音訊截短時，tts._synth_unit 會自動切半重合成（不打網路）。

背景（2026-09-27）：Gemini TTS 單次輸出的音訊長度有上限，長文會被**截斷**
（實測 2000 字只出 104.7s＝18 字/秒，根本不可能）。守門靠「合成後字/秒 ≫ 目標」
偵測截斷，自動對半再切重合成。
"""

import subprocess

from voice_dispatch import tts
from voice_dispatch.config import Config


def _make_audio(path: str, secs: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=24000:cl=mono", "-t", f"{secs:.3f}", path],
        check=True)


def _cfg() -> Config:
    c = Config()
    c.tts.speak_cps = 5.0
    return c


def test_no_split_when_pace_normal(tmp_path, monkeypatch):
    """合成結果語速正常 → 不該多切。"""
    calls = []

    def fake(text, out_path, _cfg, engine, logger=None):
        calls.append(text)
        _make_audio(out_path, len(text) / 5.0)      # 剛好符合目標語速

    monkeypatch.setattr(tts, "_synth_one", fake)
    out = str(tmp_path / "u.mp3")
    text = "字" * 200
    tts._synth_unit(text, out, _cfg(), "edge")
    assert calls == [text]


def test_split_when_model_truncates(tmp_path, monkeypatch):
    """模擬「模型輸出上限 12 秒」→ 200 字只給 12s（16.7 字/秒）→ 自動切半救回。"""
    calls = []

    def fake(text, out_path, _cfg, engine, logger=None):
        calls.append(text)
        _make_audio(out_path, min(len(text) / 5.0, 12.0))

    def fake_concat(parts, out_path, logger=None):
        _make_audio(out_path, 12.0 * len(parts))

    monkeypatch.setattr(tts, "_synth_one", fake)
    monkeypatch.setattr(tts, "_concat_audio", fake_concat)

    out = str(tmp_path / "u.mp3")
    text = "字" * 200
    tts._synth_unit(text, out, _cfg(), "edge")

    # 第 1 次整段（被截斷）→ 切成兩半各合成一次
    assert len(calls) == 3
    assert calls[0] == text
    assert calls[1] + calls[2] == text
    assert abs(tts._duration(out) - 24.0) < 1.0     # 兩段合併後 24s → 8.3 字/秒，合理


def test_split_in_half_prefers_sentence_boundary():
    a, b = tts._split_in_half_at_sentence("第一句話。第二句話。第三句話。")
    assert a.endswith("。") and b and a + b == "第一句話。第二句話。第三句話。"


def test_split_in_half_hard_split_without_punctuation():
    text = "啊" * 41
    a, b = tts._split_in_half_at_sentence(text)
    assert a + b == text
    assert len(a) == 20 and len(b) == 21
