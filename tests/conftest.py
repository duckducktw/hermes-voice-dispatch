"""★ 測試一律不准真的出聲（2026-09-26 事故）。

`tests/test_dispatch_guard.py` 直接呼叫 `d.prompt_and_capture()`（看不出來會播音），
它內部會走 `_cue_start()` / `_cue_end()` → `tts.play_chime()` → **真的**用 ffplay
把「咚咚」播到喇叭上。那個檔只 mock 了錄音與 STT，**沒有 mock 播音**。

後果：跑一次 pytest = 3 個測試 × 2 顆（cue=start + cue=end）= **6 顆咚**。
pytest 又超常被跑（每個 agent session、curator 的自我檢查都會跑），
使用者整天聽到「一次六七聲、一次會有兩個聲音」的莫名提示音，
而 daemon 自己的 log 一次都沒響 → 白查了好幾輪。**測試只驗邏輯，不該出聲。**

修法：整個測試套件掛一個 autouse fixture，把真正 spawn 播放器的那一層
（`audio._play_file_inner`，chime／beep／TTS 三條路都經過它）換成 no-op。
刻意**保留 `play_file` 本體** → 半雙工的播放狀態（`output_busy` /
`output_quiet_sec`）在測試裡仍然照常運作。
"""
import pytest

from voice_dispatch import audio


@pytest.fixture(autouse=True)
def _never_play_real_audio(monkeypatch):
    """把「真的開播放器」這層停掉；測試不會發出任何聲音。"""
    monkeypatch.setattr(audio, "_play_file_inner", lambda *a, **k: True)
