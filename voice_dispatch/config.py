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
    #   "kws" = 現行 Vosk 串接式 KWS（預設，保留既有行為）
    #   "openwakeword" = 自訂 hey_hermes ONNX 神經網路 KWS
    #   "clap" = 舊做法：拍手兩下 → 錄一段 → STT 比對字串（慢、易幻覺，只留作退路）
    mode: str = "kws"
    # ── mode="openwakeword" ──────────────────────────────────────
    # Hermes 內建的已訓練 hey_hermes 模型；路徑可由 YAML 覆寫。
    oww_model: str = "~/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx"
    oww_threshold: float = 0.6
    oww_confirmation_frames: int = 3
    # 0 = 關閉；大於 0 時交由 openWakeWord 內建 Silero VAD 閘控。
    oww_vad_threshold: float = 0.0
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
    # ══════════════════════════════════════════════════════════════════════
    #  串接式喚醒（2026-09-26 使用者定案：極限精準 + 極限延遲）
    #  第一階段「hey」閘門（常開、極省）→ 第二階段「hermes」確認（精度關）。
    #  使用者原話：「能先抓 hey，然後抓 hermes」。
    # ══════════════════════════════════════════════════════════════════════

    # ── 第一階段：hey 閘門（限制詞彙、常開、最低延遲）──────────────────────
    # 只認「hey」（+ 同音 hay）。限制詞彙解碼天生會把近似音硬解成唯一候選，
    # 所以**單獨**用誤判率高（實測對 hey 開頭的近似音 ~8.5%）——但當閘門沒關係，
    # 因為精度由第二階段負責，這裡只要「不漏掉真的 hey」。
    vosk_words: List[str] = field(default_factory=lambda: ["hey", "hay"])
    vosk_min_conf: float = 0.5
    # partial（未定案結果）也算命中 = 最早觸發、延遲最低。既然後面有確認關，
    # 這裡可以放心用粗的 partial。設 False = 只用定案結果（保守、慢一點）。
    gate_partial: bool = True

    # ── 第二階段：hermes 確認（全詞彙解碼 + bigram 規則 = 精度關）───────────
    # 為什麼用「全詞彙」：限制詞彙會把 hermit/mess 硬解成 hermes，光靠 confidence
    # 擋不掉；全詞彙解碼才看得出實際講的是 "hey hermit" 還是 "hey hermes"。
    verify_enabled: bool = True
    # 空 = 沿用 vosk_model（同顆模型：零額外記憶體、零載入延遲）。實測 664 句
    # 近似發音語料誤判 0/600、確認僅 ~40ms。要更強可指定更大的模型路徑。
    verify_model: str = ""
    # 觸發後回看的音訊長度（秒）：hey 前的 preroll + hey 之後收到的音訊。
    preroll_sec: float = 1.0
    # 從觸發起到放棄前，最多再收多少秒（去裡面找 hermes）。
    # 2.0s 讓「hey …（停頓）… hermes」有約 1.8s 的預算；實測加大不會增加誤判。
    verify_window_sec: float = 2.0
    # 收訊期間每隔多久試判一次：越小越早醒、越吃 CPU（每次確認 ~40ms）。
    # 實測最大喚醒延遲：0.15→108ms、0.2→172ms、0.3→300ms（中位數都遠早於語音結束）。
    verify_interval_sec: float = 0.15
    # 接受規則＝轉錄中出現 (前綴)(變體) 相鄰 bigram。"Hermes" 常被聽成 homes/hums，
    # "hey" 開頭也常被聽成 a/the，故兩側放寬；否決力來自變體集合本身
    # （hermit/mess/miss/mouse/mom/harm 皆不在其中 → 否決）。
    verify_prefixes: List[str] = field(default_factory=lambda: [
        "hey", "hay", "a", "the", "he", "ok", "okay", "hi",
    ])
    verify_variants: List[str] = field(default_factory=lambda: [
        "hermes", "homes", "hums", "hermis", "hermès", "hermes's",
    ])
    # 2026-09-26 由 0.5 降到 0.3（使用者「一直無法呼叫到語音助手」的實證修正）：
    #   當天 log 出現「確認轉錄 'hey hermes hey'」卻被否決 → 文字已經對了，是被這個
    #   confidence 門檻擋掉（受限詞彙/遠場收音時，正確的 hermes 常只有 0.3~0.45）。
    #   **否決力其實來自 `verify_variants` 集合本身，不是這個門檻**：微評測 44 句
    #   （正樣本 12／危險負樣本 44，含 hey hermit / her mess / Hey her mouse / hay her mess…）
    #   在 0.5 / 0.35 / 0.3 / 0.0 四段門檻下 **FP 都是 0/44**（唯一漏判是印度口音
    #   被全詞彙聽成 'he hands'，那是文字層問題、降門檻救不到）。
    #   → 降門檻純賺 recall、量測不到 FP 代價。要更保守就把它加回 0.5。
    verify_min_conf: float = 0.3       # 全詞彙 per-word 信心度門檻
    # ── 半雙工：我們自己在出聲時不做喚醒偵測 ──────────────────────
    # 2026-09-26 使用者回報「說完需求後，過一下子會連響好幾聲咚咚」。
    # 根因：語音回報的 TTS 在**背景 thread** 播（daemon.py `_watch_thread_for_result`），
    # 主迴圈同時間已經回到 wait_for_wake() 在聽 → 把自己的聲音（提示音/TTS）收進來
    # → 誤喚醒 → 播一輪提示音 → 又收回來 → 連響；遠場（隔著桌面）尤其明顯。
    # 修法＝播放期間完全不餵音訊給喚醒偵測器，播完再等 echo_guard_sec 讓殘響／
    # 裝置緩衝排掉才恢復監聽。設 0 ＝關掉半雙工（回到舊行為）。
    echo_guard_sec: float = 0.8
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
    preroll_timeout_sec: float = 3.0     # 前置靜音等待上限（都沒講話就放棄；2026-09-26 8→3s）
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
    # 完全沒收到錄音（前置靜音逾時）時要不要重試？
    # 2026-09-25 使用者定案：「錄音結束**或沒收到錄音**就翻過來咚懂，其他時間不要有」
    # → 沒收到就結束這一輪（只響「開始一聲＋結束一聲」），不要在那裡反覆重問
    #   （否則一次靜默會變成 4 輪 × 2 聲 = 8 聲提示，使用者明確說不要）。
    retry_on_no_speech: bool = False
    agree_words: List[str] = field(default_factory=lambda: [
        "對", "是", "好", "沒錯", "正確", "可以", "ok", "go", "嗯", "yes",
    ])
    disagree_words: List[str] = field(default_factory=lambda: [
        "不", "錯", "重講", "再說", "重來", "no",
    ])


@dataclass
class TtsConfig:
    # 合成引擎："edge"（edge-tts，免費快但合成腔明顯）／"gemini"（Gemini TTS，更像真人）。
    engine: str = "edge"
    # voice 兩引擎共用：engine="edge" 時是 edge 聲音名（zh-CN-XiaoxiaoNeural）；
    # engine="gemini" 時是 Gemini prebuilt 聲音名（Charon／Orus／Alnilam／Algenib…）。
    voice: str = "zh-CN-XiaoyiNeural"
    # ── engine="gemini" 專用（2026-09-26 使用者：「我要更像真人，那種商業大佬的感覺」）──
    # 風格指示：**必須**用「# 風格指示 / # 台詞」分節寫進 prompt，模型才會只唸台詞
    # （寫成「請用…口吻說出：」會被連指示一起唸出來，實測 5.3s → 14.1s）。
    gemini_style: str = (
        "你是一位五十歲的集團總裁，正在對全公司高層宣布決議。"
        "聲音低沉、有壓場感、收尾果斷。"
        "不要播報腔、不要活潑、不要上揚。"
    )
    # 精準語速旋鈕（合成後 atempo 後處理，**不變調**；兩引擎通用）。1.0＝原速。
    # 2026-09-26 使用者聽完 Charon 樣本（5.72s）：「這個，稍微快點」。
    # 為什麼不用風格指示控速度：實測「語速稍快」與「語速偏快」的輸出逐位元相同
    # （18860 bytes / 4.6s）→ prompt 對速度幾乎沒有可調性。所以速度只認這裡。
    speed: float = 1.0
    # 語速**正規化**（優先用這個；0＝停用、改用上面的 speed）。
    # 實測同一 voice 在不同 model 語速差很多（3.1→4.6s／3.8→5.72s／lite→6.64s），
    # 而配額用完會自動換 model → 固定倍率會讓語速忽快忽慢。設成「每秒幾個字」
    # 就跨 model 一致：使用者聽到的 5.72s/26 字 ≈ 4.55 字/秒，要「稍微快點」→ 5.2。
    speak_cps: float = 5.2
    # 品質守門門檻：合成後「字/秒」低於此值＝Gemini 很可能把風格指示也唸出來了
    # （實測正常 4.6~6.6s / 26 字 ≈ 4~5.6 字/秒；唸出指示會變 1.7 字/秒）→ 去掉指示重合成。
    gemini_min_cps: float = 3.0
    # 免費層**每個 model 每天只有 10 次**配額 → 主 model 用完就依序換 fallback。
    gemini_model: str = "gemini-3.1-flash-tts-preview"
    gemini_model_fallbacks: List[str] = field(default_factory=lambda: [
        "gemini-3.8-flash-tts",
        "gemini-3.8-flash-lite-tts",
        "gemini-2.5-flash-preview-tts",
    ])
    gemini_api_key_env: str = "GOOGLE_API_KEY"
    gemini_env_file: str = "~/.hermes/.env"
    gemini_timeout_sec: float = 60.0
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
    # 提示模式（2026-09-25 使用者要求）：
    #   "chime"（預設）＝整個互動只用「咚咚」兩聲，不講話：
    #       喚醒 → 咚（tone1）咚（tone2）；聽完需求（有講／沒講都算）→ 再一次咚咚。
    #   "voice"＝舊行為，用 edge-tts 講 ok_prompt / retry_prompts / dispatched_prompt。
    prompt_mode: str = "chime"
    # 派工完成後用「語音」把結果念出來（使用者 2026-09-26 要求）。
    # 只念一句短摘要（`spoken_summary` 去掉 Markdown、截斷），完整內容仍在 Discord 串內。
    speak_result: bool = True
    speak_result_max_chars: int = 160
    # relay 模式：盯討論串、等它安靜幾秒就把最後一則 agent 訊息念出來。
    # 2026-09-26 使用者：「TTS 合成加速，現在跑完還要等一下才能聽到」——
    # 元凶就是這個安靜判定原本 60s（實測 16:54 回覆、16:55:08 才出聲＝慢 66s）。
    # 降到 8s：串流結束後幾乎立刻念，又還留有餘裕避免唸到半截的訊息。
    speak_result_quiet_sec: float = 8.0
    # 盯串的輪詢間隔（越小越快發現新訊息，越吃 API 額度）。
    speak_result_poll_sec: float = 2.0
    # relay 模式判斷「回合真的結束」用的 Hermes session DB。判斷依據（實測）：
    #   回合結束 = 最後一則 role=assistant/finish_reason='stop' 的訊息
    #              ＋緊接一筆 role=session_meta 收尾列。
    # 只有看到新的 session_meta 才念最終回覆 → 絕不唸中間訊息、回合結束立刻出聲。
    speak_result_session_db: str = "~/.hermes/state.db"
    # 咚咚＝「有質感的」雙音提示。
    # 兩次退貨記錄（2026-09-25）：
    #   1. 純正弦 + 衰減 → 「空洞的咚咚」
    #   2. 泛音列 + 雜訊敲擊瞬態（木琴/木魚/太鼓）→ 「這他媽是火車」
    # 定案方向：**蘋果（iOS/macOS）那種感覺** = FM 調變鈴聲
    #   （Glass / Marimba / Tri-tone / Note 都是 FM 合成，乾淨、明亮、無雜訊敲擊）。
    #   y = sin(2πft + I(t)·sin(2π·f·R·t))，I 在數十毫秒內收乾 → 亮起音 + 純淨尾韻。
    # 參數全部可調，預設＝Apple「Glass」感（非整數調變比 → 鐘／玻璃）。
    chime_tone1_hz: float = 1174.66  # D6（第一聲「咚」）
    chime_tone2_hz: float = 880.0    # A5（第二聲「咚」，下行五度）
    chime_tone_dur_sec: float = 0.10
    chime_gap_sec: float = 0.08      # 兩聲起音的間隔
    chime_volume: float = 0.32
    # FM 調變（蘋果系音色的核心）。fm_ratio=0 就退回純加法合成。
    chime_fm_ratio: float = 1.41     # 非整數 → 玻璃/鐘；整數 1.0→馬林巴、2.0→清脆
    chime_fm_index: float = 3.5      # 調變深度（越大起音越亮）
    chime_fm_decay_sec: float = 0.045  # 指數收乾時間（蘋果音「乾淨」的關鍵）
    # 加法泛音層（補厚度；蘋果系只留基頻，[比值, 振幅, 衰減倍率]）
    chime_partials: List[List[float]] = field(default_factory=lambda: [
        [1.00, 1.00, 1.00],
    ])
    chime_decay_sec: float = 0.35    # 基頻的衰減時間常數
    chime_tail_sec: float = 0.55     # 尾韻長度（樂音自然收尾 + 殘響）
    chime_detune_cents: float = 0.0  # 第二層微失諧（拍頻）；蘋果系乾淨 → 0
    chime_attack_noise: float = 0.0  # 敲擊雜訊比例；蘋果系不要開（會變木頭／火車）
    chime_reverb: float = 0.35       # 殘響濕度（0 = 乾扁的電腦音）
    chime_reverb_taps: List[List[float]] = field(default_factory=lambda: [
        [0.031, 0.30],               # [延遲秒數, 相對音量]
        [0.057, 0.20],
        [0.089, 0.12],
    ])
    # 音源：`"files"`＝播原廠素材音檔（**有素材就用這個**）；`"synth"`＝合成退路。
    # 使用者 2026-09-25 兩次退貨合成版後定案：「你找找蘋果素材」
    # → 用蘋果原廠音效（macOS aiff / iOS tones），存在
    #   ~/.local/share/hermes-voice-dispatch/chime/raw/（**不入 repo，版權**）
    chime_source: str = "synth"
    # 要播的音檔（依序串接；單一檔案＝整顆 cue）。檔案不存在就自動退回合成。
    chime_files: List[str] = field(default_factory=list)
    # 開頭／結尾**各掛不同音**（使用者 2026-09-25：「一聲高一聲低，像 Discord 開關 mic，
    # 但不要一樣，我會搞錯」）。留空＝退回上面的 chime_files（兩邊同一顆）。
    chime_start_files: List[str] = field(default_factory=list)
    chime_end_files: List[str] = field(default_factory=list)
    chime_file_max_sec: float = 2.5  # 單一素材最長取用秒數（避免提示音太長）


@dataclass
class SttConfig:
    # STT 一律經由此包裝腳本（cgroup 保護，避免 OOM），回傳一行 JSON
    scoped_script: str = "~/.hermes/scripts/voice_task/stt_scoped.sh"
    # 用「預先量化好的 int8」模型而非原始 fp16：ctranslate2 對 fp16 模型每次載入都要
    # 現場量化，實測 large-v3 載入 17.9s（GPU 峰值 93%）＝使用者說的「說一句卡一次」。
    # ellisd/faster-whisper-large-v3-int8 是同一個 large-v3 的 int8 CT2 版，載入只要
    # 2.2s、VRAM 更低，而且對 14 則真實語音轉錄**逐字完全相同**（2026-09-26 實測）。
    # 換掉不影響辨識品質，只是不再每次講話都重載/重量化整個模型。
    model: str = "ellisd/faster-whisper-large-v3-int8"
    # 推論裝置："auto"（預設＝有 CUDA 走 GPU）／"cuda"／"cpu"。
    device: str = "auto"
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
    # 串首訊息（＝發在母頻道、同時是討論串起點那一則）。
    # 2026-09-26 使用者定案：「只留需求那段」「語音任務不用特地說」→ **就是需求原文
    # 本身**，不加標題、不加 emoji、不加時間（外面看起來跟打字下的任務一樣）。
    # 這則同時是討論串開頭（thread_id == message_id），gateway 開 session 會把它帶進
    # 上下文，所以需求的完整原文全場只出現一次（串內 relay 只送指向句，見 relay.mode）。
    card_template: str = "{transcript}"
    # 自動把使用者加入討論串（2026-09-26 使用者：「自動把我加到串裏面」）。
    # 語音派工的討論串是 bot 開的（不是由使用者自己的訊息長出來的），所以他預設
    # 不在成員名單裡、收不到通知。建立後用
    #   PUT /channels/{thread}/thread-members/{user_id}
    # 把他加進去。空字串＝不加入。填 Discord user id。
    user_id: str = ""
    # 討論串名稱：直接用需求前幾字（同一天要求：不做自動化痕跡，不加「🎙️」）。
    thread_name_template: str = "{short}"
    thread_short_len: int = 40                 # thread 名取轉錄前幾字
    # 串內再補一則任務卡（**留空＝不發**；預設不發，避免洗版）。
    task_card_template: str = ""
    # 派工通知（**留空＝不發**；gateway 接手後自己會回，這則是多餘的）。
    dispatched_notice: str = ""
    # token 來源
    env_file: str = "~/.hermes/.env"
    token_env: str = "DISCORD_BOT_TOKEN"
    timeout_sec: float = 30.0


@dataclass
class DispatchConfig:
    hermes_bin: str = "hermes"
    # 2026-09-26 使用者反饋「開啟的串處理速度慢，幾乎不做事」：原因=中轉每次 API 呼叫
    # 7~9s × 多步工具呼叫（實測一支燈光任務做了 24 次呼叫 ≈ 3 分鐘），加上舊提示詞
    # 叫它「需要確認就停下來等」。改成：直接動手、少探索、回報精簡、非必要不反問。
    prompt_template: str = (
        "你剛收到一則使用者用「語音」下的需求（語音派工）。需求原文：\n"
        "「{transcript}」\n"
        "\n"
        "直接動手完成，並把結果回報到這條 Discord 討論串：\n"
        "  channel_id={channel_id}  thread_id={thread_id}\n"
        "回報指令：hermes send --to discord:{channel_id}:{thread_id} \"<訊息>\"\n"
        "\n"
        "要求（重要）：\n"
        "1. 立刻做，不要等待、不要為了確認而反問；只有真的缺關鍵資訊才在串內問一句後停。\n"
        "2. **你做的每一步都要讓使用者在串內看得到**——否則他會以為你沒在做事。\n"
        "   開工先發一句；之後每完成一個主要步驟就發一行短訊息（做了什麼／結果）；\n"
        "   不要連續沉默太久。\n"
        "3. 只載入真正相關的技能，不要探索無關檔案、也不要讀 daemon 自己的技能"
        "（voice-task-pipeline／hermes-voice-dispatch 與你無關）。\n"
        "4. **最後一定要發一則結論**（做了什麼＋怎麼驗證；失敗也發原因）——"
        "沒發這則 = 沒完成。訊息短即可，但別省掉。"
    )
    # 派工子程序的 log 目錄
    log_dir: str = "~/.local/state/hermes-voice-dispatch/dispatch"
    # 心跳：派工還在跑時，每幾秒在串內發一則「仍在處理中」。0 = 關閉。
    # （2026-09-26 使用者：「有做事但都沒輸出到 dc，看起來就沒有」——長任務中間
    #  沉默太久會讓人以為沒動。心跳讓「有在做事」看得見。）
    heartbeat_sec: float = 120.0


@dataclass
class RelayConfig:
    """把語音需求「當成一則使用者訊息」送進 Discord 討論串 → gateway 用它原本那條路處理。

    為什麼改成這樣（2026-09-26 使用者定案）：
      「要從 gateway 處理語音出來的任務。像是我打字→開串→gateway 處理，只是把打字變成語音。」
    gateway 只處理使用者訊息、會忽略機器人訊息；但 Hermes 的 Discord adapter 原生支援
    **接受受信任 bot 的訊息**（`DISCORD_ALLOW_BOTS=mentions`：只接受 @提及 Hermes 的
    bot 訊息，官方文件明講這是給 relay／webhook bot 用的）。所以：
      daemon 用一個 **Discord webhook**（author 是 webhook 自己、`bot=true`、id≠gateway bot）
      把「<@gateway_bot> 需求原文」發進討論串 → gateway 視為一般訊息 → 同一條 session、
      同樣的 typing/逐字串流/後續追問接續＝跟打字完全一樣。

    與舊做法的差別：不必自己 spawn `hermes -z`、也不必自組 dispatch prompt；agent 就是
    這條討論串的 agent。前提：gateway 的 `.env` 要有 `DISCORD_ALLOW_BOTS=mentions`，
    且 `streaming.enabled=true`（逐字）。
    """
    enabled: bool = True
    env_file: str = "~/.hermes/.env"
    # 完整 webhook URL（含 token）。空＝停用，回退舊的 spawn `hermes -z`。
    url_env: str = "DISCORD_RELAY_WEBHOOK_URL"
    # gateway bot 的 user id：訊息內容要 @ 它，gateway 才會收（ALLOW_BOTS=mentions）。
    mention_id: str = "1520796555580543138"
    # webhook 發話時顯示的名稱。**留空＝模仿 Hermes bot 自己**（daemon 用
    # `GET /users/@me` 取 bot 的名字＋頭像；2026-09-26 使用者：「要模仿 bot 在伺服器
    # 的外觀」）。填了就用填的字串，但頭像仍會用 bot 的。
    username: str = ""
    timeout_sec: float = 20.0


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
    relay: RelayConfig = field(default_factory=RelayConfig)
    # 主 log 檔（daemon 生命週期用）
    log_file: str = "~/.local/state/hermes-voice-dispatch/dispatch.log"

    # 執行期填入，不從 YAML 讀
    discord_token: Optional[str] = None
    relay_url: Optional[str] = None

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
        # relay webhook（含 token）也放 .env；空字串＝relay 停用，回退 spawn。
        cfg.relay_url = read_env_value(
            expand(cfg.relay.env_file), cfg.relay.url_env
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
