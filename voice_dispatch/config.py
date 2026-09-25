"""設定載入：dataclass 預設值 + YAML 覆寫 + ~/.hermes/.env 解析。

所有可調參數（門檻、id、路徑、指令）都集中在這裡，程式其他地方一律讀 config，
不寫死任何頻道 id / token / 路徑 / 門檻。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    # 輸入裝置：可給 sounddevice 的 index(int) 或裝置名稱子字串(str)；
    # None = 系統預設來源。
    # 2026-09-25 定案：本機（acer-ubuntu）**內建擷取路徑全部失效**——Mic1/Mic2 與
    # 所有 ALSA capture 裝置都是「開檔約 1 秒後凍結成直流」的罐頭緩衝（上游
    # thesofproject/sof#11216，軟體層已掃完無解）。故改用手機麥克風：
    # `PhoneMic` 是 phone-mic-bridge 建出的虛擬來源（scrcpy --audio-source=
    # mic-unprocessed → null sink → remap），bridge 偵測到有人開麥時自動拉起。
    # 釘死裝置名，不依賴會被 WirePlumber 改動的系統預設。
    device: Optional[Union[int, str]] = "PhoneMic"
    # 凍結偵測 / 自動恢復（本機 DMIC 會「開檔後立刻凍結」，見 README 疑難排解）
    frozen_max_blocks: int = 240     # 連續多少個位元相同的區塊視為凍結（240×64ms≈15s）
    # 凍結時的「破壞性」恢復指令。**預設關閉（空清單）**：本機 mic 凍結是 SOF DMIC
    # 驅動層問題（linux sof#11216），重啟 PipeWire 只換到約 15 秒的正常訊號就又凍結，
    # 但 restart pipewire/pipewire-pulse 會把「所有 app」的音訊串流一起砍掉
    # （2026-09-25 實測：每 2 分鐘一次、一天 36 次 → Discord / Minecraft 音訊反覆斷線）。
    # 要回復舊行為就把 systemctl 那行填回來（建議只在確定沒其他 app 在用音訊時開）。
    recover_command: List[str] = field(default_factory=list)
    recover_cooldown_sec: float = 120.0   # 兩次恢復之間最短間隔
    recover_wait_sec: float = 7.0         # 恢復後等裝置回來
    # 連續健康幾秒才把「恢復失敗計數」歸零。重啟後 mic 常會假活十幾秒就又凍結，
    # 若一通過就歸零，退避永遠長不起來（會變成每 2 分鐘重啟一次）。
    recover_reset_healthy_sec: float = 180.0
    status_file: str = "~/.local/state/hermes-voice-dispatch/mic-status.json"
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
    abs_floor: float = 0.035        # RMS 絕對下限，低於此一律不算瞬變（避免安靜環境誤判）
    # 2026-09-25 實測校準（手機麥克風、PhoneMic 增益 150%）：
    #   環境底噪 rms 中位數 0.00063→加增益後 0.00275
    #   使用者真實拍手區塊 rms 只有 0.021~0.042（三組都一樣）
    #   → 原本 0.05 的門檻**全部卡掉**，拍手永遠不觸發（症狀：喊了好幾下沒反應）。
    #   phone-mic-bridge 已把手機 mic 增益拉到 150%（≈+10.6dB ≈ 3.4 倍），
    #   拍手因此落在 0.071~0.142；門檻降到 0.035 留 2 倍餘裕，
    #   而底噪 0.00275 距門檻仍有 ~13 倍，不會誤觸。
    abs_ceil: float = 1.2           # RMS 絕對上限，超過視為破音/雜訊，不算拍手
    min_gap_sec: float = 0.12       # 兩次拍手最短間隔
    max_gap_sec: float = 1.5        # 兩次拍手最長間隔
    refractory_sec: float = 0.08    # 一次瞬變後的不反應期，避免同一拍手被算兩次


@dataclass
class WakeConfig:
    # 喚醒方式：
    #   "kws"  = openWakeWord 神經網路關鍵詞模型（真實助手做法，推薦）
    #   "clap" = 舊做法：拍手兩下 → 錄一段 → STT 比對字串（慢、易幻覺，只留作退路）
    mode: str = "kws"
    # 可選：hey_jarvis / alexa / hey_mycroft / hey_rhasspy / timer / weather。
    # 想要「Hermes」需另外訓練自訂模型（openWakeWord 目前只支援英文喚醒詞）。
    kws_models: List[str] = field(default_factory=lambda: ["hey_jarvis"])
    kws_threshold: float = 0.5
    # 喚醒引擎：
    #   "vosk" (預設) = Vosk 限制詞彙解碼當關鍵詞偵測。**支援任意英文詞**
    #       （所以能用「hermes」），免訓練、免 AccessKey。實測單次 0.05~0.08s，
    #       對 4 段真實房間背景（共 56 秒）零誤觸。
    #   "openwakeword" = 預訓練模型，不支援自訂詞（只有 hey_jarvis 等）。
    kws_engine: str = "vosk"
    vosk_model: str = "~/.local/share/hermes-voice-dispatch/vosk-model-small-en-us-0.15"
    # 兩詞詞彙表比單詞表好：實測正例 5/5（單詞表只有 4/5，某個聲音漏判），
    # 也避免「只有一個詞」時任何像語音的東西被強制對上。
    # 信心度門檻 0.8 可擋掉近似音（實測 "The hurries of modern life"
    # conf=0.58~0.69 被擋掉）；真同音詞（"Her mess…" conf=1.0）擋不掉，屬正常。
    vosk_words: List[str] = field(default_factory=lambda: ["hermes", "hey hermes"])
    # 太敏感 → 0.8 調到 0.9（2026-09-25 使用者反饋「太敏感了」）。
    # 實測正例（含台灣腔）幾乎都 1.0，所以 0.9 不會漏判，但能擋掉更多近似音。
    vosk_min_conf: float = 0.9
    # ── mode="clap"（舊路徑）─────────────────────────────────────
    window_sec: float = 2.5
    cooldown_sec: float = 10.0
    keywords: List[str] = field(default_factory=lambda: [
        "hermes", "赫米斯", "赫密斯", "哈米斯",
    ])


@dataclass
class VadConfig:
    # 2026-09-25 校準（手機麥克風 + PhoneMic 增益 150%）：
    # 環境底噪 rms 實測 0.00275。原本 0.02 是照筆電近場 mic 訂的，
    # 隔著桌面收音時你的聲音常常過不了 → VAD 太晚才開始（切掉字頭）
    # 或直接判定沒人講話 → 程式就一直在那裡重問需求。
    # 降到 0.012：距底噪還有 ~4 倍，但抓得到小聲的起頭。
    speech_rms_threshold: float = 0.012  # 判定為語音的 RMS 門檻（use_silero=False 時才用）
    # 2026-09-25：改用 Silero 神經網路 VAD 當預設端點偵測（見 kws.py）。
    # RMS 門檻怎麼調都是蹺蹺板：太敏感會把冷氣／風扇當語音，太保守則隔著桌面
    # 講話被判成「沒人講話」→ 程式一直重問需求。Silero 是訓練過的小模型，穩健得多。
    use_silero: bool = True
    silero_threshold: float = 0.5
    # 省資源：安靜時（RMS 低於此）直接跳過 Silero 推論。實測環境底噪 ~0.0028，
    # 所以 0.006 不會漏掉說話起頭，但能讓待機時幾乎不耗 CPU。
    silero_rms_gate: float = 0.006
    preroll_keep_sec: float = 0.5        # 語音起點前額外保留的音訊（避免切掉字頭）
    # 2026-09-25：錄需求前先排掉串流緩衝。TTS 播放期間我們不讀串流，
    # PortAudio/pulse 會把那段音訊積在緩衝；不排掉的話，一開始錄就會先讀到
    # **我們自己剛剛講的話**，VAD 把它當成使用者需求 → STT → 派工假任務
    # （實測 log 的需求原文 = 它自己的台詞「這次說大聲一點。」「我在聽。」）。
    settle_sec: float = 1.2
    preroll_timeout_sec: float = 8.0     # 前置靜音等待上限（都沒講話就放棄）
    trailing_silence_sec: float = 1.2    # 講完後連續靜音多久視為結束
    max_record_sec: float = 60.0         # 單次錄音最長
    min_record_sec: float = 0.5          # 單次錄音最短（低於此不算數）


@dataclass
class ConfirmConfig:
    # ⚠️ 2026-09-25 起確認輪不再阻塞（fail-open），以下三個欄位目前**不再生效**，
    # 只留著避免舊設定檔載入時出錯：agree_words（不再需要說「對」）、
    # record_sec（不再另外錄一段確認音）、max_unclear（不再有 unknown 迴圈）。
    # 真正還在用的只有 max_retries（需求重錄上限）。
    record_sec: float = 3.0              # 回述確認時錄音長度
    # 2026-09-25：3 → 2。使用者反饋「一直卡在問我需求」——重試太多次會讓它
    # 像壞掉的答錄機。現在最多問 3 次（第 1 次 + 2 次重問），而且每次換句話說。
    max_retries: int = 2                 # 需求重錄上限
    max_unclear: int = 2                 # 連續無法判定的上限
    agree_words: List[str] = field(default_factory=lambda: [
        "對", "是", "好", "沒錯", "正確", "可以", "ok", "go", "嗯", "yes",
    ])
    disagree_words: List[str] = field(default_factory=lambda: [
        "不", "錯", "重講", "再說", "重來", "no",
    ])


@dataclass
class TtsConfig:
    voice: str = "zh-CN-XiaoyiNeural"
    # 2026-09-25：文案改成「像在跟人講話」——短、口語、而且**每次重問換一句**，
    # 不要像機器人一樣重播同一句（使用者反饋「一直卡在問我需求」＋「不夠自然」）。
    # 再一輪（使用者：「好愛說廢話，說話快一點，不要重複我的內容」）：
    #   - rate=+30% 讓它講快一點
    #   - confirm_template 清空＝**不再複述使用者的內容**（那只是在重複他剛講的話）
    #   - 所有提示語再縮短
    rate: str = "+30%"              # edge-tts 語速（使用者要求「說話快一點」）
    ok_prompt: str = "我在聽。"
    retry_prompts: List[str] = field(default_factory=lambda: [
        "再說一次？",
        "大聲一點？",
        "沒收到，再說一次？",
    ])
    # 複述：**刻意留空＝不複述**（只在需要告知時才講話，別重複使用者內容）
    confirm_template: str = ""
    unclear_prompt: str = "再說一次。"
    give_up_prompt: str = "先這樣。"
    dispatched_prompt: str = "好。"
    # 靜音模式：存在這個檔就完全不合成、不播放 TTS（測試時用，不必重啟 daemon）。
    #   touch ~/.local/state/hermes-voice-dispatch/mute   → 靜音
    #   rm    ~/.local/state/hermes-voice-dispatch/mute   → 恢復
    mute_file: str = "~/.local/state/hermes-voice-dispatch/mute"
    beep_enabled: bool = False      # 機器感的「嗶」預設關掉（TTS 本身就是提示）
    beep_freq: float = 880.0
    beep_dur_sec: float = 0.18
    beep_volume: float = 0.3


@dataclass
class SttConfig:
    # STT 一律經由此包裝腳本（cgroup 保護，避免 OOM），回傳一行 JSON
    scoped_script: str = "~/.hermes/scripts/voice_task/stt_scoped.sh"
    model: str = ""                 # 空字串 = 用腳本預設模型（large-v3）
    # 喚醒詞只用來比對「hermes」，不需要 large-v3。實測單次耗時：
    # large-v3 7.0s / small 2.6s / base 1.9s（皆有認出 Hermes）。
    # 喚醒詞是冷啟動路徑，每省 4 秒都很有感，所以單獨用快模型。
    wake_model: str = "Systran/faster-whisper-small"
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
