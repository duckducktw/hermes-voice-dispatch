"""設定載入：dataclass 預設值 + YAML 覆寫 + ~/.hermes/.env 解析。

所有可調參數（門檻、id、路徑、指令）都集中在這裡，程式其他地方一律讀 config，
不寫死任何頻道 id / token / 路徑 / 門檻。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - venv 內一定有 PyYAML
    yaml = None


def expand(path: str) -> str:
    """展開 ~ 與環境變數。"""
    return os.path.expandvars(os.path.expanduser(path))


# --------------------------------------------------------------------------
# 各區段設定
# --------------------------------------------------------------------------
@dataclass
class AudioConfig:
    samplerate: int = 16000          # 取樣率（Hz）
    channels: int = 1               # 單聲道
    blocksize: int = 1024           # 每個讀取區塊的樣本數
    device: Optional[int] = None    # 輸入裝置 index；None = 系統預設
    # 播放器候選，依序嘗試（會用 shlex 拆成 argv，{file} 由檔名取代）
    players: List[str] = field(default_factory=lambda: [
        "ffplay -nodisp -autoexit -loglevel quiet {file}",
        "paplay {file}",
        "aplay -q {file}",
    ])


@dataclass
class ClapConfig:
    rms_window_sec: float = 5.0      # 自適應門檻的觀察視窗（取中位數）
    threshold_mult: float = 4.0     # RMS 需超過「近期中位數 × 此倍數」
    abs_floor: float = 0.05         # RMS 絕對下限，低於此一律不算瞬變（避免安靜環境誤判）
    abs_ceil: float = 1.2           # RMS 絕對上限，超過視為破音/雜訊，不算拍手
    min_gap_sec: float = 0.12       # 兩次拍手最短間隔
    max_gap_sec: float = 1.5        # 兩次拍手最長間隔
    refractory_sec: float = 0.08    # 一次瞬變後的不反應期，避免同一拍手被算兩次


@dataclass
class WakeConfig:
    window_sec: float = 2.5         # 拍手後錄多久去比對喚醒詞
    cooldown_sec: float = 10.0      # 喚醒詞沒過之後的冷卻，避免誤觸時反覆載入 STT
    keywords: List[str] = field(default_factory=lambda: [
        "hermes", "赫米斯", "赫密斯", "哈米斯",
    ])


@dataclass
class VadConfig:
    speech_rms_threshold: float = 0.02   # 判定為語音的 RMS 門檻
    preroll_timeout_sec: float = 6.0     # 前置靜音等待上限（都沒講話就放棄）
    trailing_silence_sec: float = 1.2    # 講完後連續靜音多久視為結束
    max_record_sec: float = 60.0         # 單次錄音最長
    min_record_sec: float = 0.5          # 單次錄音最短（低於此不算數）


@dataclass
class ConfirmConfig:
    record_sec: float = 3.0              # 回述確認時錄音長度
    max_retries: int = 3                 # 需求重錄上限
    max_unclear: int = 2                 # 連續無法判定的上限
    agree_words: List[str] = field(default_factory=lambda: [
        "對", "是", "好", "沒錯", "正確", "可以", "ok", "go", "嗯", "yes",
    ])
    disagree_words: List[str] = field(default_factory=lambda: [
        "不", "錯", "重講", "再說", "重來", "no",
    ])


@dataclass
class TtsConfig:
    voice: str = "zh-TW-HsiaoChenNeural"
    ok_prompt: str = "OK，請說出你的需求。"
    confirm_template: str = "我理解成：{transcript}。對嗎？"
    unclear_prompt: str = "抱歉我沒聽清楚，請再說一次要或不要。"
    give_up_prompt: str = "我先放棄，請再說一次。"
    dispatched_prompt: str = "好的，已經派工出去了。"
    # beep 提示音（用 numpy 合成正弦波，不需外部音檔）
    beep_freq: float = 880.0
    beep_dur_sec: float = 0.18
    beep_volume: float = 0.3


@dataclass
class SttConfig:
    # STT 一律經由此包裝腳本（cgroup 保護，避免 OOM），回傳一行 JSON
    scoped_script: str = "~/.hermes/scripts/voice_task/stt_scoped.sh"
    model: str = ""                 # 空字串 = 用腳本預設模型
    timeout_sec: float = 180.0
    ffmpeg_bin: str = "ffmpeg"      # 前處理：正規化成 16k mono wav


@dataclass
class DiscordConfig:
    api_base: str = "https://discord.com/api/v10"
    channel_id: str = "1392008000197365870"   # #人工智障
    guild_id: str = "1340241728959025236"
    auto_archive_duration: int = 1440          # thread 自動封存（分鐘）
    thread_name_limit: int = 100               # Discord thread 名長度上限
    card_template: str = "🎙️ \"{transcript}\""
    thread_name_template: str = "🎙️ {short}"
    thread_short_len: int = 40                 # thread 名取轉錄前幾字
    task_card_template: str = (
        "**🎙️ 語音任務**\n"
        "需求原文：\n"
        "> {transcript}\n"
        "\n"
        "建立時間：{date}"
    )
    dispatched_notice: str = "✅ 收到，開始處理。"
    # token 來源
    env_file: str = "~/.hermes/.env"
    token_env: str = "DISCORD_BOT_TOKEN"
    timeout_sec: float = 30.0


@dataclass
class DispatchConfig:
    hermes_bin: str = "hermes"
    prompt_template: str = (
        "你剛收到一則使用者用「語音」下的需求（語音派工）。需求原文：\n"
        "「{transcript}」\n"
        "\n"
        "請把這件事當成一般任務處理，並把過程與結果完整回報到這條 Discord 討論串：\n"
        "  channel_id={channel_id}  thread_id={thread_id}\n"
        "回報指令：hermes send --to discord:{channel_id}:{thread_id} \"<訊息>\"\n"
        "\n"
        "步驟：\n"
        "1. 先在串內發一則開工訊息，說明你打算怎麼做。\n"
        "2. 執行過程中把重要進度發到串內（不要沉默太久）。\n"
        "3. 若中途需要使用者確認，就在串內發問並停下來等；使用者會直接在串內回覆你。\n"
        "4. 完成後把結果發到串內，包含你實際做了什麼、怎麼驗證的。若失敗也要回報失敗原因。\n"
    )
    # 派工子程序的 log 目錄
    log_dir: str = "~/.local/state/hermes-voice-dispatch/dispatch"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    clap: ClapConfig = field(default_factory=ClapConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    confirm: ConfirmConfig = field(default_factory=ConfirmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    # 主 log 檔（daemon 生命週期用）
    log_file: str = "~/.local/state/hermes-voice-dispatch/dispatch.log"

    # 執行期填入，不從 YAML 讀
    discord_token: Optional[str] = None

    # ------------------------------------------------------------------
    def resolved_scoped_script(self) -> str:
        return expand(self.stt.scoped_script)

    def resolved_log_file(self) -> str:
        return expand(self.log_file)

    def resolved_dispatch_log_dir(self) -> str:
        return expand(self.dispatch.log_dir)


# --------------------------------------------------------------------------
# 合併邏輯：把 dict（來自 YAML）套用到 dataclass 之上
# --------------------------------------------------------------------------
def _apply_overrides(obj: Any, data: Dict[str, Any]) -> None:
    """就地把 data 內的鍵套到 dataclass 實例 obj 上，未知鍵忽略。"""
    if not isinstance(data, dict):
        raise ValueError(f"設定區段應為對應表（mapping），實得：{type(data).__name__}")
    valid = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            # 未知鍵：略過但不報錯，方便向前相容
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str] = None, *, load_token: bool = True) -> Config:
    """載入設定。

    path 為 None 時只用預設值（再加上 .env 的 token）。
    """
    cfg = Config()

    if path:
        p = Path(expand(path))
        if not p.exists():
            raise FileNotFoundError(f"找不到設定檔：{p}")
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML 未安裝，無法讀取 YAML 設定")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"設定檔頂層應為對應表：{p}")
        _apply_overrides(cfg, raw)

    if load_token:
        cfg.discord_token = read_env_value(
            expand(cfg.discord.env_file), cfg.discord.token_env
        )

    return cfg


# --------------------------------------------------------------------------
# .env 解析（不引入 python-dotenv，手工解析即可）
# --------------------------------------------------------------------------
def parse_env_file(path: str) -> Dict[str, str]:
    """解析簡單的 KEY=VALUE .env 檔，回傳 dict。檔案不存在則回傳空 dict。"""
    result: Dict[str, str] = {}
    p = Path(expand(path))
    if not p.exists():
        return result
    for raw_line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # 去掉成對的引號
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def read_env_value(path: str, key: str) -> Optional[str]:
    """從 .env 取單一鍵；優先使用行程環境變數。"""
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    return parse_env_file(path).get(key)
