"""設定測試：預設值、YAML 覆寫、.env 解析。"""

import os

import pytest

from voice_dispatch.config import (
    Config,
    load_config,
    parse_env_file,
    read_env_value,
)


def test_defaults():
    cfg = load_config(None, load_token=False)
    assert cfg.discord.channel_id == "1392008000197365870"
    assert cfg.discord.guild_id == "1340241728959025236"
    assert cfg.audio.samplerate == 16000
    assert cfg.clap.threshold_mult == 4.0
    assert "hermes" in cfg.wake.keywords
    assert cfg.discord.auto_archive_duration == 1440
    assert cfg.discord.thread_name_limit == 100


def test_yaml_override(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "audio:\n"
        "  samplerate: 8000\n"
        "  device: 3\n"
        "discord:\n"
        "  channel_id: \"999\"\n"
        "clap:\n"
        "  threshold_mult: 6.5\n",
        encoding="utf-8",
    )
    cfg = load_config(str(p), load_token=False)
    assert cfg.audio.samplerate == 8000
    assert cfg.audio.device == 3
    assert cfg.discord.channel_id == "999"
    assert cfg.clap.threshold_mult == 6.5
    # 未覆寫的鍵維持預設
    assert cfg.audio.channels == 1
    assert cfg.discord.guild_id == "1340241728959025236"


def test_yaml_unknown_key_ignored(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("audio:\n  bogus_key: 123\n  samplerate: 22050\n", encoding="utf-8")
    cfg = load_config(str(p), load_token=False)
    assert cfg.audio.samplerate == 22050
    assert not hasattr(cfg.audio, "bogus_key")


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/xyz.yaml", load_token=False)


def test_parse_env_file(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# 註解\n"
        "\n"
        "DISCORD_BOT_TOKEN=abc123\n"
        "export FOO='hello world'\n"
        'BAR="quoted"\n'
        "NOEQ\n",
        encoding="utf-8",
    )
    env = parse_env_file(str(p))
    assert env["DISCORD_BOT_TOKEN"] == "abc123"
    assert env["FOO"] == "hello world"
    assert env["BAR"] == "quoted"
    assert "NOEQ" not in env


def test_parse_env_missing_file(tmp_path):
    assert parse_env_file(str(tmp_path / "nope.env")) == {}


def test_read_env_value_prefers_os_environ(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("MYKEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("MYKEY", "from_env")
    assert read_env_value(str(p), "MYKEY") == "from_env"
    monkeypatch.delenv("MYKEY", raising=False)
    assert read_env_value(str(p), "MYKEY") == "from_file"


def test_load_config_reads_token(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DISCORD_BOT_TOKEN=tok_from_file\n", encoding="utf-8")
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text(
        f"discord:\n  env_file: \"{env}\"\n", encoding="utf-8"
    )
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    cfg = load_config(str(cfgfile), load_token=True)
    assert cfg.discord_token == "tok_from_file"
