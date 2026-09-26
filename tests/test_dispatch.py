"""dispatch：最終輸出抽取（保底回報用）＋心跳看護。"""
import subprocess

from voice_dispatch import dispatch


def test_extract_stdout_strips_header(tmp_path):
    p = tmp_path / "d.log"
    p.write_text(
        "# dispatch spawned 2026-09-26T00:00:00\n"
        "# argv=['hermes', '-z', 'x']\n"
        "\n"
        "完成，已回報。\n",
        encoding="utf-8",
    )
    assert dispatch.extract_stdout(str(p)) == "完成，已回報。"


def test_extract_stdout_header_only_is_empty(tmp_path):
    p = tmp_path / "d.log"
    p.write_text("# dispatch spawned 2026-09-26T00:00:00\n# argv=['hermes']\n",
                 encoding="utf-8")
    assert dispatch.extract_stdout(str(p)) == ""


def test_extract_stdout_missing_file():
    assert dispatch.extract_stdout("/nonexistent/nope.log") == ""


class _FakeProc:
    """會先丟 N 次 TimeoutExpired、之後才結束的假程序。"""
    def __init__(self, timeouts: int):
        self.timeouts = timeouts
        self.calls = 0

    def wait(self, timeout=None):
        self.calls += 1
        if self.calls <= self.timeouts:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        return 0


def test_watch_heartbeats_then_exit():
    """心跳：程序還活著時每輪回呼一次，結束後才呼叫 on_exit。"""
    events = []
    proc = _FakeProc(timeouts=3)
    dispatch._watch(
        proc, "/tmp/x.log",
        lambda p, l: events.append("exit"),
        lambda s: events.append(("hb", s)),
        1.0,
    )
    assert events[-1] == "exit"
    assert sum(1 for e in events if isinstance(e, tuple)) == 3


def test_watch_without_heartbeat_just_waits():
    events = []
    dispatch._watch(_FakeProc(timeouts=0), "/tmp/x.log",
                    lambda p, l: events.append("exit"), None, 0.0)
    assert events == ["exit"]
