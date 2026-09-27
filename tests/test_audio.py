"""audio.resolve_device 的裝置名比對測試（不需要真的 PortAudio）。"""
import os

from voice_dispatch import audio
from tests.conftest import REAL_PLAY_FILE_INNER

P = "alsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic"
DEVS = [
    {"name": "pulse", "max_input_channels": 32},
    {"name": P + ".HiFi__Mic1__source", "max_input_channels": 2},
    {"name": P + ".HiFi__Mic2__source", "max_input_channels": 2},
]


class _FakeSD:
    def __init__(self, devices=DEVS):
        self._devices = devices

    def query_devices(self, device=None):
        if device is None:
            return self._devices
        try:
            return self._devices[int(device)]
        except (TypeError, ValueError):
            pass
        for d in self._devices:
            if d["name"] == device:
                return d
        raise ValueError(f"No input device matching '{device}'")


def test_none_and_int_pass_through():
    sd = _FakeSD()
    assert audio.resolve_device(sd, None) is None
    assert audio.resolve_device(sd, 1) == 1


def test_exact_name_kept():
    assert audio.resolve_device(_FakeSD(), "pulse") == "pulse"


def test_short_name_matches_full_portaudio_name():
    sd = _FakeSD()
    assert audio.resolve_device(sd, "HiFi__Mic1__source") == 1
    assert audio.resolve_device(sd, "HiFi__Mic2__source") == 2


def test_unknown_name_returned_unchanged():
    assert audio.resolve_device(_FakeSD(), "no-such-mic") == "no-such-mic"


def test_output_only_node_not_matched():
    sd = _FakeSD([{"name": "x.HiFi__Mic1__source.monitor", "max_input_channels": 0}])
    assert audio.resolve_device(sd, "HiFi__Mic1__source") == "HiFi__Mic1__source"


# ── 2026-09-28 播放端爆音回歸（使用者第三次回報「還是會」才查到的真因）──────

def test_playback_timeout_scales_with_duration(tmp_path):
    """播放逾時不能是固定值，否則長語音播到一半被砍。

    真因：原本寫死 timeout=60，95 秒的長回報播到第 60 秒被 SIGKILL →
    被當成「播放器失敗」→ 退到 aplay 播 mp3 → 全振幅白噪音。
    """
    import subprocess as sp
    long_mp3 = str(tmp_path / "long.mp3")
    sp.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=f=440:d=95", "-b:a", "128k", long_mp3], check=True)
    assert audio._playback_timeout(long_mp3) > 95.0

    short = str(tmp_path / "s.wav")
    sp.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=f=440:d=1", short], check=True)
    assert audio._playback_timeout(short) >= 60.0      # 保留下限


def test_never_feeds_mp3_to_pcm_only_player(tmp_path, monkeypatch):
    """mp3 絕不能餵給 aplay：它會把壓縮位元組當樣本播出去＝白噪音。"""
    import subprocess as sp
    from voice_dispatch.config import AudioConfig

    mp3 = str(tmp_path / "a.mp3")
    sp.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=f=440:d=1", mp3], check=True)

    tried = []
    real = sp.run

    def spy(argv, **kw):
        name = argv[0] if isinstance(argv, list) else str(argv)
        if "ffprobe" in name:
            return real(argv, **kw)
        tried.append(os.path.basename(name))

        class R:
            returncode = 1
            stderr = b""
        return R()

    monkeypatch.setattr(audio.subprocess, "run", spy)
    cfg = AudioConfig(players=["ffplay {file}", "aplay -q {file}"])
    REAL_PLAY_FILE_INNER(mp3, cfg, logger=None)
    assert "aplay" not in tried, "mp3 被餵給 aplay → 會爆出白噪音"


def test_timeout_does_not_fall_through_to_worse_player(tmp_path, monkeypatch):
    """逾時＝播到一半被砍，不是格式不支援 → 不該再退到別的播放器重播一次。"""
    import subprocess as sp
    from voice_dispatch.config import AudioConfig

    mp3 = str(tmp_path / "b.mp3")
    sp.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", "sine=f=440:d=2", mp3], check=True)

    tried = []
    real = sp.run

    def spy(argv, **kw):
        name = argv[0] if isinstance(argv, list) else str(argv)
        if "ffprobe" in name:
            return real(argv, **kw)
        tried.append(os.path.basename(name))
        raise sp.TimeoutExpired(argv, 60)

    monkeypatch.setattr(audio.subprocess, "run", spy)
    cfg = AudioConfig(players=["ffplay {file}", "mpv {file}", "ffmpeg {file}"])
    assert REAL_PLAY_FILE_INNER(mp3, cfg, logger=None) is False
    assert tried == ["ffplay"], f"逾時後還往下退階：{tried}"


def test_default_players_have_no_pcm_only_entry():
    """預設播放器清單不能再出現 aplay（爆音來源）。"""
    from voice_dispatch.config import AudioConfig
    for tmpl in AudioConfig().players:
        first = os.path.basename(tmpl.split()[0])
        assert first not in audio._PCM_ONLY_PLAYERS, f"預設清單含 PCM-only 播放器：{tmpl}"
