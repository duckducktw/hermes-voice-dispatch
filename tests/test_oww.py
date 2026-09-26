"""openWakeWord 自訂 hey_hermes 模型測試（不需要麥克風）。"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from voice_dispatch.oww import OwwSpotter, OwwUnavailable

MODEL = Path("~/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx").expanduser()
CORPUS = Path("/tmp/wake_corpus")


def _require_model_and_corpus() -> Path:
    if not MODEL.is_file():
        pytest.skip("需要本機 hey_hermes openWakeWord 模型")
    labels = CORPUS / "labels.jsonl"
    if not labels.is_file():
        pytest.skip("需要 /tmp/wake_corpus 真實語料")
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    positive = next((row for row in rows if row["label"] == 1), None)
    if positive is None:
        pytest.skip("wake corpus 沒有正樣本")
    wav = CORPUS / positive["file"]
    if not wav.is_file():
        pytest.skip(f"找不到正樣本：{wav}")
    return wav


def _read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


def test_hey_hermes_corpus_audio_triggers():
    """真實語料中的 Hey Hermes 應通過 0.6 / 連續三幀確認。"""
    samples = _read_pcm16(_require_model_and_corpus())
    spotter = OwwSpotter(str(MODEL), threshold=0.6, confirmation_frames=3)
    assert any(spotter.feed(samples[i:i + 1024]) for i in range(0, len(samples), 1024))


def test_silence_and_noise_do_not_trigger():
    """靜音與中等強度白雜訊不可觸發。"""
    if not MODEL.is_file():
        pytest.skip("需要本機 hey_hermes openWakeWord 模型")
    spotter = OwwSpotter(str(MODEL), threshold=0.6, confirmation_frames=3)
    silence = np.zeros(16000, dtype=np.int16)
    assert not any(spotter.feed(silence[i:i + 1024]) for i in range(0, len(silence), 1024))
    spotter.reset()
    rng = np.random.default_rng(20260926)
    noise = (rng.normal(0, 500, 16000)).astype(np.int16)
    assert not any(spotter.feed(noise[i:i + 1024]) for i in range(0, len(noise), 1024))


def test_missing_model_is_reported(tmp_path):
    """模型缺失時讓 daemon 可捕捉 OwwUnavailable 並回退，不在初始化時 crash。"""
    with pytest.raises(OwwUnavailable):
        OwwSpotter(str(tmp_path / "missing.onnx"), 0.6, 3)
