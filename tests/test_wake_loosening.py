"""用『實際 log 的失敗案例』驗證喚醒放寬後會命中。

2026-09-26 使用者：「讓他更寬鬆點，我剛剛叫了幾次不回」。
這些 tokens 直接抄自 daemon log 的「hey 閘門觸發但窗內未見 hermes（轉錄 …, tokens=…）」。
"""
from __future__ import annotations

from voice_dispatch.config import Config
from voice_dispatch.kws import match_wake_variants

# (轉錄, [(token, conf), ...])
OBSERVED_FAILURES = [
    ("hey harm if", [("hey", 1.0), ("harm", 1.0), ("if", 1.0)]),
    ("hey her me", [("hey", 1.0), ("her", 0.908624), ("me", 0.908624)]),
    ("hi harry", [("hi", 0.527242), ("harry", 0.235128)]),
    ("the army", [("the", 1.0), ("army", 1.0)]),
    ("hey hurries", [("hey", 1.0), ("hurries", 0.470776)]),
    ("here", [("here", 0.646346)]),
]


def _match(cfg, pairs):
    tokens = [t for t, _ in pairs]
    confs = [v for _, v in pairs]
    return match_wake_variants(
        tokens, confs, cfg.wake.verify_prefixes, cfg.wake.verify_variants,
        cfg.wake.verify_min_conf, allow_bare_variant=True,
    )


def test_observed_log_failures_now_match():
    """這 6 個都是『閘門響了卻被否決』的真實案例，放寬後必須命中。"""
    c = Config()
    for text, pairs in OBSERVED_FAILURES:
        assert _match(c, pairs), f"應該要命中：{text}"


def test_still_rejects_dangerous_lookalikes():
    """否決力要還在：hermit / mouse / mess 這些近似音不能收（否則會誤喚醒）。

    ⚠️ 已知取捨：加了變體 `her` 之後，`hey her …`（例：舊語料的負樣本
    「Hey her mouse」）**會命中** —— 這是為了救「hermes 被切成 her me」而接受的代價。
    """
    c = Config()
    for tokens, confs in (
        (["hey", "hermit"], [1.0, 1.0]),
        (["hey", "mess"], [1.0, 1.0]),
        (["okay", "hermits"], [1.0, 1.0]),
        (["hi", "mouse"], [1.0, 1.0]),
    ):
        assert match_wake_variants(
            tokens, confs, c.wake.verify_prefixes, c.wake.verify_variants,
            c.wake.verify_min_conf, allow_bare_variant=True,
        ) is None, f"不該命中：{tokens}"


def test_min_conf_lowered_but_present():
    c = Config()
    assert c.wake.verify_min_conf <= 0.15
    # 低信心但文字對（'hi harry' 的 harry 僅 0.235）→ 要過
    assert _match(c, [("hi", 0.5), ("harry", 0.235)])
