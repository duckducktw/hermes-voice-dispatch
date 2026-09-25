# SPEC — hermes-voice-dispatch

語音派工守護程式（Voice Dispatch Daemon）。跑在使用者的 Linux 筆電上，
24 小時監聽麥克風。使用者「拍手兩下 + 大喊 Hermes」即喚醒，
用語音說出需求 → 程式複述（不等待確認）→ 把需求轉發到 Discord 頻道並開討論串
→ 透過 Hermes agent 開始幹活，過程與結果都回報到該討論串。

## 使用者的原始需求（逐字，繁體）

> 幫我叫Cloud Code 寫一個軟體,讓他監聽我筆電的麥克風,只要我拍手兩下,
> 大喊一聲Hermes,他就會要有一個提示音,OK說出我的需求,然後他會先跟我
> 確定一遍需求之後,把我的內容轉發到這個人工智障的頻道,然後開啟討論串
> 開始幹活。然後如果中途有什麼需要確定,跟結果跟我報告,然後也可以讓我
> 知道,然後把他推到GitHub上面去

## 執行環境（已實測確認，不要假設）

- OS: Linux, Ubuntu GNOME, 筆電（含內建麥克風）
- 音效：PipeWire/PulseAudio，播放可用 `paplay` / `aplay` / `ffplay`；`ffmpeg` 已安裝
- 麥克風：`arecord -l` 可見 `sof-hda-dsp` 的 `HDA Analog` / `DMIC`
- Python：**一律使用 Hermes venv 的 python** → `~/.hermes/hermes-agent/venv/bin/python3`
  已裝妥：`numpy 2.4.3`、`scipy 1.17.1`、`sounddevice 0.5.5`、`faster_whisper 1.2.1`、`edge_tts 7.2.7`
  （**不要**新增 numpy/sounddevice 之外的重依賴；不要引入 torch、openai-whisper、pyaudio）
- STT 既有工具（**已存在，直接叫用，不要重寫**）：
  `~/.hermes/scripts/voice_task/stt_scoped.sh <audio_in> - [model]`
  → stdout 為一行 JSON：`{"transcript": "...", "duration_sec": 1.23, "model": "...", "device": "cuda"}`
  這一層是 systemd memory-capped scope 包裝（faster-whisper large-v3 int8，peak ~3.2GB），
  **必須經由它呼叫 STT**，否則會把 gateway OOM 掉。10 秒音檔約 10 秒。
- TTS：`edge-tts`（venv 內），中文語音建議 `zh-TW-HsiaoChenNeural`（女聲）。
  edge-tts 輸出 mp3 → 用 `ffplay -nodisp -autoexit -loglevel quiet` 播放。
- Hermes agent CLI：`hermes`（在 PATH）
  - `hermes -z "<prompt>"` = one-shot，跑一個完整 agent session，stdout 只印最終回覆
  - `hermes send --to discord:<channel_id>:<thread_id> "<text>"` = 直接發訊息到指定 Discord 討論串
    （重用 gateway 的 bot token，無需 LLM、無需 gateway 在跑）
    `hermes send --to discord:<channel_id> "<text>"` = 發到主頻道
- GitHub：`gh` 已以 `duckducktw` 登入（token 具 `repo` / `workflow` scope）

## 設定（由 user 提供，寫進 config 不要寫死）

- Discord bot token：`~/.hermes/.env` 的 `DISCORD_BOT_TOKEN`
- 目標主頻道 id（「人工智障」）：`1392008000197365870`
- Guild id：`1340241728959025236`

## 功能需求

### R1 喚醒偵測（喊「Hermes」）
**2026-09-25 架構改版**：從「雙拍手 + 大喊 Hermes → STT 比對字串」改成真實
語音助手的做法——常開跑關鍵詞偵測，**不對它做 STT**。

1. 以 `sounddevice` 開 16kHz / 單聲道 / blocksize 1024 的 InputStream 持續讀取。
2. 引擎：**Vosk 限制詞彙解碼**（`voice_dispatch/kws.py`，`wake.kws_engine="vosk"`）。
   解碼詞彙鎖成 `["hermes", "hey hermes", "[unk]"]`，等同關鍵詞偵測，免訓練、
   支援任意英文詞。只看 final result 的 per-word `conf`，須 ≥ `wake.vosk_min_conf`
   （0.8）才命中——partial 沒有信心度，會被雜音「強制」解成喚醒詞。
3. 另一引擎 `wake.kws_engine="openwakeword"`（預訓練 hey_jarvis 等）保留可切換，
   但**不支援自訂詞**。
4. 舊路徑保留成 `wake.mode="clap"` 僅作退路；預設 `wake.mode="kws"`。

#### 為什麼廢掉舊路徑（2026-09-25 實測）
- 命中後還要 2.5s 錄音 + 2.6~7s STT 才開始回應 → 使用者體感「太智障」。
  Home Assistant 官方文件的說法：「喚醒詞必須極快處理——不能讓助手在
  喚醒詞講完 5 秒後才開始聽。」
- 短音訊 + 小聲時 STT 常吐幻覺（實測得到「感謝收看。」「那個更難,那個更難」），
  喚醒詞因此漏判。
- 還要拍手，完全不像在跟人講話。
- 順帶修掉「回音自我觸發」：TTS 播放期間不讀串流，pulse 會把那段音訊積在
  緩衝，開始錄需求時先讀到的就是自己的提示語 → 被當成需求派工假任務。
  見 R2 的 `vad.settle_sec` 與 `_looks_like_own_prompt()`。

### R2 語音引導 + 錄需求
1. 喚醒成功 → 播一次提示音（`tts.prompt_mode="chime"`，預設），**不講話**；
   聽完需求（有講或沒講逾時都算）→ 再播一次。這兩聲就是使用者要的全部提示。
   - 兩聲**必須明顯不同**（2026-09-25 使用者：「一聲高一聲低，像 Discord 開關 mic，
     但不要一樣，我會搞錯」）→ 用 `tts.chime_start_files` / `tts.chime_end_files`
     各掛一顆；留空則退回共用的 `tts.chime_files`（兩邊同一顆）。
   - 音源兩種（`tts.chime_source`）：
     `"files"`＝直接播素材音檔（**正式用法**）；`"synth"`＝FM 調變合成（沒素材的機器才用）。
     合成版被使用者退貨兩次（「空洞的咚咚」→「這他媽是火車」），
     所以**結論是別自己合成，用現成素材**。
     - 素材不入 repo（版權），放 `~/.local/share/hermes-voice-dispatch/chime/`
       （`pairs/`＝高低配對、`quad/`＝敲四下、`cue_*`＝雙擊咚咚、`raw/`＝原廠原檔、
       `packs/`＝CC0 包）；本機用哪一組寫在 `config.yaml`（**已被 .gitignore 排除**，
       systemd 用 `--config` 指過去）。
     - 合成相關參數（`tts.chime_*`：FM 比／指數／泛音列／殘響…）保留但非主力。
   - 要回舊行為（TTS 講 `ok_prompt`／`retry_prompts`／`dispatched_prompt`）→
     `tts.prompt_mode="voice"`；舊的 beep（`tts.beep_enabled`）只在 voice 模式生效。
2. 進入錄音：以 **Silero 神經網路 VAD** 偵測語音起訖（`vad.use_silero` 預設 true；
   退回 RMS 門檻則用 `vad.speech_rms_threshold`）—— 前置靜音等待最多 8 秒，
   偵測到語音後，連續靜音超過 1.2 秒即停止，最長 60 秒，最短 0.5 秒。
   另含 `vad.preroll_keep_sec`（0.5s）把判定前的字頭補回來。
   `vad.settle_sec`（1.2s）會先排掉緩衝——提示音比它短，不會被錄進去。
   完全沒收到錄音（前置靜音逾時）時：預設**直接結束本輪、不再重問**
   （`confirm.retry_on_no_speech=false`）——使用者要的是「就翻過來咚懂，其他時間不要有」，
   重問會讓一次靜默變成 4 輪 × 2 聲提示音。

### R3 逐字轉錄
呼叫 `stt_scoped.sh <wav> -`，解析 JSON 的 `transcript`。
輸入音檔請先正規化成 16kHz 單聲道 wav（ffmpeg），STT 前處理失敗要有清楚錯誤。

### R4 複述（**不阻塞**，2026-09-25 改版）
1. TTS 說：「我理解成：<轉錄>，直接派工。」（`tts.confirm_template` 可設定模板）
   —— **純告知：不等待使用者回覆、不錄確認音、不做 agree/disagree 判定。**
2. 唯一的重錄條件：R3 的轉錄**整句就只是**否定／重來詞（不／錯／重來／再說，
   且正規化後 ≤ 5 字）→ TTS 說「好的，請重新說一次需求。」→ 回 R2，
   最多 `confirm.max_retries` 次。
3. **為什麼改成 fail-open**：舊版固定錄 3 秒等使用者說「對」，但「對」只有約
   0.3 秒，VAD 幾乎全砍掉 → Whisper 回吐幻覺（實測連續得到「然後想說,造孽啊。」
   「作為成功我肯定會破防。」）→ 使用者卡在鬼打牆，一次互動要一分鐘。
   `confirm.record_sec` / `confirm.agree_words` / `confirm.max_unclear` 已停用
   （保留欄位只為相容舊設定檔）。

### R5 轉發到 Discord + 開討論串
1. 用 Discord REST（bot token）發一則到主頻道，內容為語音派工卡：
   `🎙️ 語音派工｜<轉錄>`
2. 以該訊息建立 public thread（`POST /channels/{channel}/messages/{msg_id}/threads`），
   thread 名 = `🎙️ <轉錄前 20 字> MM-DD HH:mm`（需處理 Discord thread 名長度上限 100）。
3. 把完整任務卡（需求原文 + 時間 + 來源=語音）發進該討論串。

### R5b 必須「像是平常的任務一樣」（user directive 2026-09-25）
使用者明確要求：語音派工產生的討論串要**跟平常在這個頻道下的任務一模一樣**，
不是特殊格式的自動訊息。因此：
- thread 名一律用 `🎙️ 語音任務 MM-DD HH:mm`（與 gateway 既有語音任務慣例一致）。
- 頻道內那則派工訊息與串內任務卡要讀起來像人下的任務，不要出現「webhook / 自動化 / API」字樣。
- 派工出去的那個 Hermes session 必須把該討論串當成**一般任務串**在跑：
  使用者在串內任何回覆，都要能被 Hermes 正常接手（gateway 會為 thread 建 session 並帶入串內歷史），
  所以任務卡裡要把「需求原文 + 這是一般任務」交代清楚，讓後續串內對話能無縫接續。
- **不要**另外發明需要使用者學習的新指令或新格式。

### R6 派工給 Hermes 並回報
1. 背景啟動（detached，不阻塞守護迴圈）：
   `hermes -z "<dispatch_prompt>"`，stdout/stderr 存到 log 檔。
2. `dispatch_prompt` 模板（可設定）內容必須交代 agent：
   - 這是語音派工任務，需求原文在此：<轉錄>
   - 回報目標：`hermes send --to discord:<channel>:<thread>` 發到該討論串
   - 開工時先發一則開工訊息到串內
   - 執行中把重要進度發到串內；**若中途需要使用者確認，在串內發問並等候**
   - 完成後把結果/摘要發到串內
3. 守護程式本身也要在串內發一則「已派工，Hermes 開始執行」的狀態訊息。

### R7 生命週期
- 單一 daemon 程序；`--once` 模式可只跑一輪；`--simulate "<文字>"` 可跳過麥克風直接
  走 R3 之後的流程（供測試 / 無麥克風環境）；`--dry-run` 只做本地流程不發 Discord/不派工。
- SIGINT/SIGTERM 乾淨關閉（釋放音訊串流）。
- log 用 python logging，預設寫 `~/.local/state/hermes-voice-dispatch/dispatch.log`（可設定）。

## 非功能需求
- 不得在沒有麥克風或沒有音效裝置時崩潰：給明確錯誤訊息 + 非零 exit code。
- 所有外部指令（stt_scoped.sh / ffmpeg / ffplay / hermes）都經由 config 可覆寫，
  且把 stderr 收進 log 方便除錯。
- CPU 閒置時要低：喚醒偵測只做 RMS 計算（numpy），不得持續跑 STT。
- 附 `systemd/voice-dispatch.service`（user service）：`Restart=always`、
  `ExecStart=<venv python> -m voice_dispatch`、讀取 `~/.hermes/.env`（`EnvironmentFile` 可省略，
  由程式自己解析 `~/.hermes/.env` 取 token）。

## 專案結構（請照此建立）
```
hermes-voice-dispatch/
├── README.md                 # 繁體中文，含安裝、校正、systemd、疑難排解
├── CLAUDE.md                 # 給未來 agent 的專案摘要
├── SPEC.md                   # 本檔
├── requirements.txt
├── .gitignore
├── pyproject.toml            # 可選，但 console script 入口 voice-dispatch 建議有
├── config.example.yaml
├── voice_dispatch/
│   ├── __init__.py
│   ├── __main__.py           # python -m voice_dispatch
│   ├── cli.py                # argparse：--once / --simulate / --dry-run / --config / --list-devices
│   ├── config.py             # dataclass + YAML 載入 + 預設值 + ~/.hermes/.env 解析
│   ├── audio.py              # 裝置列舉、InputStream 迴圈、WAV 寫入、播放(ffplay/paplay)
│   ├── clap.py               # 雙拍手偵測（純函式，吃 numpy 區塊，可單元測試）
│   ├── vad.py                # 能量 VAD 錄音（含靜音超時）
│   ├── stt.py                # 呼叫 stt_scoped.sh、解析 JSON、正規化音檔
│   ├── tts.py                # edge-tts 合成 + 播放、合成 beep
│   ├── wake.py               # 組裝：拍手 → 喚醒詞 STT 驗證
│   ├── discord_api.py        # REST：發訊息、建 thread、錯誤處理（429/403 明確報錯）
│   ├── dispatch.py           # 組 dispatch_prompt、背景 spawn `hermes -z`
│   └── daemon.py             # 主狀態機：idle → wake → record → confirm → dispatch
├── tests/
│   ├── test_clap.py          # 合成尖峰訊號：雙拍手命中、單拍手不命中、間隔過長不命中、雜訊不誤觸
│   ├── test_vad.py           # 靜音切斷、最長上限
│   ├── test_text.py          # 喚醒詞/同意/不同意 判定、thread 名截斷
│   └── test_config.py        # 預設值、YAML 覆寫、.env 解析
└── systemd/
    └── voice-dispatch.service
```

## 驗收標準（你必須自己跑過並貼出真實輸出）
1. `~/.hermes/hermes-agent/venv/bin/python3 -m pytest -q` 全綠。
2. `voice_dispatch.py --simulate "測試需求" --dry-run` 能走完本地流程並印出將發送的內容，
   不發任何網路請求。
3. `voice_dispatch.py --list-devices` 能列出麥克風裝置。
4. `python -m compileall` 無錯誤；`ruff`（若安裝）無致命錯誤可略過。
5. 不要真的對 Discord 發送或真的 spawn hermes（測試時用 --dry-run）。

## 規範
- 繁體中文註解與 README（台灣用語）。程式碼識別字英文。
- 不要寫死任何頻道 id / token / 路徑：全部走 config，預設值可以放 config.example.yaml。
- 寧可少做也不要留半成品；沒有實作完的功能不要出現在 README 當成已完成。
