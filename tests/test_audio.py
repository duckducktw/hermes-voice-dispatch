"""audio.resolve_device 的裝置名比對測試（不需要真的 PortAudio）。"""
from voice_dispatch import audio

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
