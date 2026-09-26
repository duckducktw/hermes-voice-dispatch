"""forward_and_dispatch 的流向測試（不打真網路、不 spawn）。

2026-09-26 使用者定案（依序收斂）：
  「對話記錄不要這麼多訊息，幾則就好」
  「需求不要發兩次」「我要外面也能看到訊息」「只留需求那段」「語音任務不用特地說」
  「能直接用 webhook 發嗎？節省一則訊息，而且要模仿 bot 在伺服器的外觀」
→ relay 路徑：webhook 直接在**母頻道**發 1 則「<@Hermes> 需求原文」（模仿 bot 的
  名字＋頭像）→ gateway 的 `force_thread_channels` 自己開討論串接手
  （session key = thread id == message id）→ 需求只出現一次、外面看得到。
  另：串是 gateway 開的，使用者不在裡面 → daemon 補 PUT thread-members。
"""
from __future__ import annotations

import voice_dispatch.daemon as daemon_mod
from voice_dispatch.config import Config
from voice_dispatch.daemon import VoiceDispatcher


class _FakeClient:
    def __init__(self, cfg, logger=None):
        self.calls = []
        self.thread_in_message = {"id": "TH1"}
        self.self_user = {"id": "BOT1", "username": "hermes agent", "avatar": "abc"}

    def get_self(self):
        self.calls.append(("get_self",))
        return dict(self.self_user)

    def get_message(self, channel_id, message_id):
        self.calls.append(("get_message", channel_id, message_id))
        return {"id": message_id, "thread": self.thread_in_message}

    def delete_message(self, channel_id, message_id):
        self.calls.append(("delete_message", channel_id, message_id))
        return {}

    def post_message(self, channel_id, content):
        self.calls.append(("post_message", channel_id, content))
        return {"id": "MSG-cmd"}

    def create_thread_from_message(self, channel_id, message_id, name,
                                   auto_archive_duration=None):
        self.calls.append(("create_thread_from_message", channel_id, message_id, name))
        return {"id": "TH-cmd"}

    def add_thread_member(self, thread_id, user_id):
        self.calls.append(("add_thread_member", thread_id, user_id))
        return {}

    def post_thread_message(self, thread_id, content):
        self.calls.append(("post_thread_message", thread_id, content))
        return {"id": "M"}

    def get_messages(self, channel_id, limit=10):
        return []


def _make(monkeypatch, relay=True, relay_result=None, relay_raises=False):
    cfg = Config()
    cfg.relay.enabled = relay
    cfg.discord.user_id = "1058750638760149033"
    cfg.relay_url = "https://discord.com/api/v10/webhooks/1/tok" if relay else ""
    fake = _FakeClient(cfg)
    monkeypatch.setattr(daemon_mod, "DiscordClient", lambda *a, **k: fake)

    sent = []

    def _post_relay(transcript, c, logger=None, identity=None):
        if relay_raises:
            raise RuntimeError("boom")
        sent.append({"transcript": transcript, "identity": identity})
        if relay_result is not None:
            return relay_result
        return {"id": "MSG1", "thread": {"id": "TH1"}}

    monkeypatch.setattr(daemon_mod.dispatch, "post_relay", _post_relay)

    spawned = []
    monkeypatch.setattr(
        daemon_mod.dispatch, "spawn_hermes",
        lambda *a, **k: spawned.append(a) or 4242,
    )

    d = VoiceDispatcher(cfg)
    d._stop = True   # 不要讓回報 watcher 真的跑起來（會打 DB／網路）
    return d, fake, sent, spawned


def test_relay_posts_single_message_and_no_self_thread(monkeypatch):
    """全場只有 1 則：由 webhook 發在母頻道；daemon 不自己發卡、不自己開串。"""
    d, fake, sent, spawned = _make(monkeypatch)
    d.forward_and_dispatch("幫我把伺服器重啟")

    assert len(sent) == 1
    assert sent[0]["transcript"] == "幫我把伺服器重啟"
    assert not any(c[0] == "post_message" for c in fake.calls)
    assert not any(c[0] == "create_thread_from_message" for c in fake.calls)
    assert spawned == [], "relay 成功時不該 spawn"


def test_relay_mimics_bot_identity(monkeypatch):
    """要模仿 bot 在伺服器的外觀：名字＋頭像都取自 GET /users/@me。"""
    d, _, sent, _ = _make(monkeypatch)
    d.forward_and_dispatch("幫我把伺服器重啟")
    ident = sent[0]["identity"]
    assert ident["username"] == "hermes agent"
    assert ident["avatar_url"] == "https://cdn.discordapp.com/avatars/BOT1/abc.png?size=128"


def test_relay_adds_user_to_gateway_thread(monkeypatch):
    """串是 gateway 開的 → 使用者不在成員名單，daemon 要補加。"""
    d, fake, _, _ = _make(monkeypatch)
    d.forward_and_dispatch("幫我把伺服器重啟")
    assert ("add_thread_member", "TH1", "1058750638760149033") in fake.calls


def test_relay_waits_for_thread_when_response_has_none(monkeypatch):
    """post_relay 回應還沒掛 thread 時，要輪詢訊息等 gateway 開串。"""
    d, fake, _, _ = _make(monkeypatch, relay_result={"id": "MSG1"})
    fake.thread_in_message = {"id": "TH9"}
    d.forward_and_dispatch("幫我把伺服器重啟")
    assert any(c[0] == "get_message" for c in fake.calls)
    assert ("add_thread_member", "TH9", "1058750638760149033") in fake.calls


def test_relay_failure_falls_back_to_spawn(monkeypatch):
    """relay 掛掉 → 回退舊路徑：bot 發卡、自行開串、spawn hermes。"""
    d, fake, sent, spawned = _make(monkeypatch, relay_raises=True)
    d.forward_and_dispatch("幫我把伺服器重啟")
    assert not sent
    assert any(c[0] == "post_message" for c in fake.calls)
    assert any(c[0] == "create_thread_from_message" for c in fake.calls)
    assert ("add_thread_member", "TH-cmd", "1058750638760149033") in fake.calls
    assert spawned, "relay 失敗要回退 spawn hermes"


def test_card_and_thread_name_have_no_voice_label():
    """「只留需求那段」「語音任務不用特地說」：沒有標題、emoji、時間。"""
    cfg = Config()
    assert cfg.discord.card_template == "{transcript}"
    assert cfg.discord.thread_name_template == "{short}"
    assert "🎙" not in cfg.discord.card_template
    assert "🎙" not in cfg.discord.thread_name_template
    assert "語音" not in dispatch_content_probe(cfg)


def dispatch_content_probe(cfg):
    """需求內容本身不該被加上任何說明文字。"""
    from voice_dispatch import dispatch
    return dispatch.build_relay_content("幫我把伺服器重啟", cfg)


def test_relay_content_has_requirement_verbatim(monkeypatch):
    d, _, sent, _ = _make(monkeypatch)
    d.forward_and_dispatch("幫我把伺服器重啟")
    from voice_dispatch import dispatch
    content = dispatch.build_relay_content("幫我把伺服器重啟", d.cfg)
    assert content.endswith("幫我把伺服器重啟")
    assert "語音" not in content


def test_dry_run_shows_single_message_and_no_extra_notice(monkeypatch, capsys):
    d, _, _, _ = _make(monkeypatch)
    d.dry_run = True
    d.forward_and_dispatch("幫我把伺服器重啟")
    out = capsys.readouterr().out
    assert "<@1520796555580543138> 幫我把伺服器重啟" in out
    assert "留空＝不發" in out


def test_relay_can_drop_the_mention(monkeypatch):
    """`mention_id` 留空＝完全不帶 @（2026-09-26 使用者要求「能改成不 @嗎」）。

    前提：gateway 的 `.env` 已把 `DISCORD_ALLOW_BOTS` 改成非 `none`/`mentions` 的值
    （本機已改成 `all`），否則 gateway 對 bot 訊息仍要求 @。
    """
    from voice_dispatch import dispatch as _d

    d, _, _, _ = _make(monkeypatch)
    d.cfg.relay.mention_id = ""
    d.forward_and_dispatch("幫我把伺服器重啟")
    content = _d.build_relay_content("幫我把伺服器重啟", d.cfg)
    assert content == "幫我把伺服器重啟"
    assert "<@" not in content


def test_relay_without_thread_falls_back_and_cleans_up(monkeypatch):
    """gateway 沒把訊息開成討論串（例如 @ 閘沒過）→ 收回孤兒訊息並回退 spawn。"""
    d, fake, _, spawned = _make(monkeypatch, relay_result={"id": "MSG1"})
    monkeypatch.setattr(d, "_await_thread", lambda *a, **k: "")
    d.forward_and_dispatch("幫我把伺服器重啟")

    assert ("delete_message", "1392008000197365870", "MSG1") in fake.calls, \
        "沒人接手的那則要收回"
    assert any(c[0] == "post_message" for c in fake.calls), "要回退到自行開串"
    assert any(c[0] == "create_thread_from_message" for c in fake.calls)
    assert spawned, "回退路徑要 spawn hermes"
