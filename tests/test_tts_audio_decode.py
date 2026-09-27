"""TTS 音訊解碼守門的回歸測試。

為什麼有這支測試（2026-09-27 使用者回報：語音回報「唸到一半突然超大聲沙」）：
`_gemini_write_audio` 舊版只用 RIFF magic 判斷格式，非 RIFF 一律當成
24kHz s16le 裸 PCM。但免費層每 model 每天只有 10 次配額，長文分段合成時會
自動輪替 model，而不同 model 回的容器／取樣率不一樣 → 把壓縮位元組當 PCM 解
就是全振幅白噪音，只有換到 model 的那一段爆掉。
"""
import os
import subprocess

import pytest

from voice_dispatch import tts


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + list(args), check=True)


def _have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


needs_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="需要 ffmpeg")


@pytest.mark.parametrize("mime,expected", [
    ("audio/L16;codec=pcm;rate=16000", 16000),
    ("audio/L16;codec=pcm;rate=24000", 24000),
    ("audio/L16; codec=pcm; rate = 22050", 22050),
    ("audio/wav", 24000),           # 沒寫 rate → 預設
    ("", 24000),
    ("audio/L16;rate=999999", 24000),   # 超出合理範圍 → 預設
    ("audio/L16;rate=10", 24000),
])
def test_pcm_rate_from_mime(mime, expected):
    assert tts._pcm_rate_from_mime(mime) == expected


@needs_ffmpeg
def test_sniff_container_detects_real_formats(tmp_path):
    mp3, wav, pcm = (str(tmp_path / n) for n in ("a.mp3", "a.wav", "a.pcm"))
    _ff("-f", "lavfi", "-i", "sine=f=440:d=1", mp3)
    _ff("-f", "lavfi", "-i", "sine=f=440:d=1", wav)
    _ff("-f", "lavfi", "-i", "sine=f=440:d=1",
        "-f", "s16le", "-ar", "24000", "-ac", "1", pcm)

    assert tts._sniff_container(open(mp3, "rb").read()) is not None
    assert tts._sniff_container(open(wav, "rb").read()) is not None
    # 裸 PCM 認不出容器 → 必須回 None 才會走 mimeType 的取樣率
    assert tts._sniff_container(open(pcm, "rb").read()) is None


@needs_ffmpeg
def test_mp3_mislabelled_as_wav_does_not_produce_noise(tmp_path):
    """核心回歸：MP3 位元組但 mimeType 寫 audio/wav，不能解成白噪音。"""
    src = str(tmp_path / "src.mp3")
    _ff("-f", "lavfi", "-i", "sine=f=440:d=1", src)
    data = open(src, "rb").read()

    out = str(tmp_path / "out.mp3")
    tts._gemini_write_audio(data, out, mime="audio/wav")
    assert os.path.getsize(out) > 0
    assert not tts._is_noise_burst(out), "容器嗅探失效 → 又會爆音"

    # 對照組：模擬舊行為（硬當裸 PCM）必須被守門抓到，否則測試本身沒意義
    raw = str(tmp_path / "old.pcm")
    old = str(tmp_path / "old.mp3")
    open(raw, "wb").write(data)
    _ff("-f", "s16le", "-ar", "24000", "-ac", "1", "-i", raw, old)
    assert tts._is_noise_burst(old), "守門門檻太緊，抓不到真正的爆音"


@needs_ffmpeg
def test_raw_pcm_uses_rate_from_mime(tmp_path):
    """16kHz 裸 PCM 要照 mimeType 解，長度才正確（舊碼會變 1.5 倍快）。"""
    pcm = str(tmp_path / "r16.pcm")
    _ff("-f", "lavfi", "-i", "sine=f=300:d=3",
        "-f", "s16le", "-ar", "16000", "-ac", "1", pcm)
    data = open(pcm, "rb").read()

    good = str(tmp_path / "good.mp3")
    tts._gemini_write_audio(data, good, mime="audio/L16;codec=pcm;rate=16000")
    assert tts._duration(good) == pytest.approx(3.0, abs=0.2)

    bad = str(tmp_path / "bad.mp3")
    tts._gemini_write_audio(data, bad, mime="audio/wav")   # 退回 24000
    assert tts._duration(bad) == pytest.approx(2.0, abs=0.2)


@needs_ffmpeg
def test_noise_burst_detects_white_noise_but_not_tone(tmp_path):
    noise = str(tmp_path / "noise.mp3")
    _ff("-f", "lavfi", "-i", "anoisesrc=d=2:a=0.99:c=white",
        "-ar", "24000", "-ac", "1", noise)
    assert tts._is_noise_burst(noise)

    # 正常訊號（峰值遠離 0 dBFS）不該被判為雜訊
    tone = str(tmp_path / "tone.mp3")
    _ff("-f", "lavfi", "-i", "sine=f=440:d=1", tone)
    assert not tts._is_noise_burst(tone)


@pytest.mark.parametrize("label,vmax,vmean,expect_noise", [
    # 實測數值（2026-09-27，ffmpeg volumedetect）——別憑感覺改這張表
    ("MP3 當裸 PCM 解（真正的 bug）", 0.0, -6.6, True),
    ("白噪音 a=0.99", 0.0, -9.3, True),
    ("真人語音 原始", -5.3, -23.1, False),
    ("真人語音 推到滿刻度", 0.0, -14.3, False),
    ("正弦波（峰值遠離滿刻度）", -18.5, -21.5, False),
])
def test_verdict_noise_matches_measured_reality(label, vmax, vmean, expect_noise):
    """用實錄 dB 數值鎖住門檻。

    合成訊號（sine+tremolo）的 crest 只有 6~7 dB，無法代理真人語音的動態，
    所以「不誤殺大聲語音」這條只能靠實測數值驗證。
    """
    assert tts._verdict_noise(vmax, vmean) is expect_noise, label


def test_noise_thresholds_sit_between_burst_and_speech():
    """門檻必須夾在「爆音 9.3」與「真人語音 14.3」之間，否則守門失效或誤殺。"""
    assert 9.3 < tts.NOISE_CREST_DB < 14.3


# ── 2026-09-27 第二輪（使用者回報「還是會」爆音後補的兩個真 bug）────────────

def test_edge_fallback_never_uses_gemini_voice():
    """Gemini 配額用完退回 edge-tts 時，不能把 Gemini 聲音名傳給 edge_tts。

    實測：edge_tts 收到 "Charon" 會 `ValueError: Invalid voice 'Charon'`，
    整段合成直接被打掉 → **備援路徑等於不存在**。
    """
    class _T:
        voice = "Charon"
        edge_voice = ""

    class _C:
        tts = _T()

    picked = tts._edge_voice(_C())
    assert picked not in tts.GEMINI_VOICES
    assert "-" in picked          # edge 的聲音名是 locale 式（zh-TW-XxxNeural）


def test_edge_voice_respects_explicit_and_passthrough():
    class _T:
        voice = "Charon"
        edge_voice = "zh-CN-XiaoxiaoNeural"

    class _C:
        tts = _T()
    assert tts._edge_voice(_C()) == "zh-CN-XiaoxiaoNeural"   # 明確指定優先

    class _T2:
        voice = "zh-TW-HsiaoChenNeural"
        edge_voice = ""

    class _C2:
        tts = _T2()
    assert tts._edge_voice(_C2()) == "zh-TW-HsiaoChenNeural"  # 本來就是 edge 名 → 沿用


@needs_ffmpeg
def test_limit_peaks_removes_intersample_overshoot(tmp_path):
    """限幅要把過衝壓到 0 dBFS 以下。

    實測坑：只掛 alimiter 不夠（真實 TTS 檔 +1.99 dB 只降到 +0.84 dB），因為 mp3
    有損重編碼自己會產生新的過衝 → 必須先靜態降增益再限幅。

    造素材的坑：乾淨 sine 就算 volume=6dB 也不會過衝（編碼器有餘裕，實測 -12 dB）。
    要逼出 inter-sample overshoot 得用**寬頻訊號 + 硬限幅 + 重取樣**。
    """
    import re

    def peak(p):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-i", p, "-af",
                            "astats=metadata=1:reset=0", "-f", "null", "-"],
                           capture_output=True, text=True)
        vals = [float(x) for x in re.findall(r"Peak level dB:\s*(-?[\d.]+)", r.stderr)]
        return max(vals) if vals else -99.0

    loud = str(tmp_path / "loud.mp3")
    _ff("-f", "lavfi", "-i", "anoisesrc=d=2:a=0.9:c=pink",
        "-af", "volume=12dB,alimiter=limit=1.0:level=false,atempo=1.3",
        "-ar", "24000", "-ac", "1", "-b:a", "128k", loud)

    before = peak(loud)
    tts._limit_peaks(loud)
    after = peak(loud)
    # 不論素材是否恰好過衝，限幅後都必須安全地低於 0 dBFS
    assert after <= 0.0, f"限幅後仍過衝（{after:+.2f} dB）→ 播放端會削波爆音"
    assert after < before, f"限幅沒起作用（{before:+.2f} → {after:+.2f}）"
