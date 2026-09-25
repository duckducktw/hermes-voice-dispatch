# SPEC — hermes-voice-dispatch

語音派工守護程式（Voice Dispatch Daemon）。跑在使用者的 Linux 筆電上，
24 小時監聽麥克風。使用者「拍手兩下 + 大喊 Hermes」即喚醒，
用語音說出需求 → 程式回述確認 → 把需求轉發到 Discord 頻道並開討論串
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

### R1 喚醒偵測（雙拍手 + 大喊 Hermes）
1. 以 `sounddevice` 開 16kHz / 單聲道 / blocksize 1024 的 InputStream 持續讀取。
2. **雙拍手**：偵測兩個「短促、尖銳的能量瞬變」。單一瞬變判定：
   該 block 的 RMS 超過「自適應門檻」（例如近 5 秒 RMS 中位數的倍數，預設 4 倍，
   並設一個絕對下限與上限避免誤判），且上升/下降極快（能量集中在 1–2 個 block）。
   兩個瞬變間隔落在 0.12–1.5 秒之間 → 判定雙拍手。
3. 雙拍手後進入「喚醒詞視窗」：錄最多 2.5 秒音訊，呼叫 STT，
   若轉錄文字（大小寫正規化、去標點）包含 `hermes` / `赫米斯` / `赫密斯` / `哈米斯` / `Hermes`
   任一者 → 喚醒成功。否則靜默重置（不發任何音效/訊息）。
4. 參數全部可設定，並提供「拍手門檻自動校正」的說明（見 README）。

### R2 提示音 + 語音引導
1. 喚醒成功 → 播一段短提示音（用 numpy 合成正弦 beep，不需外部音檔），
   再以 TTS 說「OK，請說出你的需求。」（`ok_prompt` 可設定）
2. 進入錄音：以能量 VAD 偵測語音結束 —— 前置靜音等待最多 6 秒，
   偵測到語音後，連續靜音超過 1.2 秒即停止，最長 60 秒，最短 0.5 秒。

### R3 逐字轉錄
呼叫 `stt_scoped.sh <wav> -`，解析 JSON 的 `transcript`。
輸入音檔請先正規化成 16kHz 單聲道 wav（ffmpeg），STT 前處理失敗要有清楚錯誤。

### R4 回述確認（必做，且可重錄）
1. TTS 說：「我理解成：<轉錄>。對嗎？」（`confirm_prompt` 可設定模板）
2. 錄 3 秒 → STT → 判定：
   - **同意**：包含 對/是/好/沒錯/正確/可以/OK/go/嗯 → 進入 R5
   - **不同意**：包含 不/錯/重講/再說/重來 → 回到 R2.2 重新錄需求（最多重試 3 次）
   - **無法判定**：再問一次；連兩次無法判定 → TTS 說「我先放棄，請再說一次」並重置
3. 語音確認失敗不應阻塞超過設定的重試上限。

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
