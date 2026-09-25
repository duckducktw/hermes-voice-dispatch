# hermes-voice-dispatch

Linux 筆電上的**語音派工守護程式**。

> 拍手兩下 👏👏 + 大喊「**Hermes**」→ 提示音 → 說出你的需求 → 程式回述跟你確認 →
> 把需求轉發到 Discord 的 `#人工智障` 頻道並開一條討論串 → 背景派工給 Hermes agent
> 執行，過程與結果都回報到那條討論串。

實作細節與逐條需求見 [`SPEC.md`](SPEC.md)；給未來 agent 的快速索引見 [`CLAUDE.md`](CLAUDE.md)。

---

## 運作流程

```
idle ──雙拍手──▶ 喚醒詞視窗(STT) ──命中「Hermes」──▶ 提示音 + 「請說出你的需求」
   ▲                                                         │
   │                                                         ▼
   └──────── 放棄/完成 ◀── 派工 hermes ◀── 轉發 Discord ◀── 回述確認（可重錄）
```

- **R1 喚醒**：`sounddevice` 以 16kHz/單聲道持續讀取，只算 RMS（CPU 幾乎閒置）。
  偵測兩個「短促尖銳的能量瞬變」（自適應門檻），間隔落在 0.12–1.5 秒即判定雙拍手；
  接著錄 2.5 秒喚醒詞視窗，經 STT 比對 `hermes / 赫米斯 / 赫密斯 / 哈米斯`，命中才喚醒。
- **R2 引導**：播一次提示音（`tts.prompt_mode="chime"`，預設，**不講話**）；
  聽完需求（有講或沒講逾時都算）再播一次。兩聲語意不同：
  **懂咚＝收到喚醒**、**咚懂＝錄音結束／沒收到錄音**，其他時間不出聲。
  素材可用 `tools/make_cues.py` 重新產生（低沉＋快＋抖動，全合成、無版權問題）：
  ```bash
  ~/.hermes/hermes-agent/venv/bin/python3 tools/make_cues.py --variant 3 --loud
  ```
- **R3 轉錄**：能量 VAD 錄音 → 正規化成 16k mono wav → 經 `stt_scoped.sh` 轉錄。
- **R4 回述確認**：TTS「我理解成：…。對嗎？」→ 錄音判定同意/不同意/無法判定，可重錄。
- **R5 轉發**：Discord REST 發語音派工卡到主頻道並開 public thread，串內補完整任務卡。
- **R6 派工**：背景 `hermes -z "<prompt>"`，並在串內貼「已派工」狀態；agent 自行用
  `hermes send --to discord:<channel>:<thread>` 回報進度與結果。

---

## 需求與依賴

**一律使用 Hermes venv 的 python**，所有依賴都已裝在裡面：

```bash
PY=~/.hermes/hermes-agent/venv/bin/python3
```

- Python 3.11、`numpy` / `sounddevice` / `edge-tts` / `faster-whisper` / `PyYAML`
- 系統工具：`ffmpeg`、`ffplay`（或 `paplay` / `aplay`）、`arecord`
- `sounddevice` 需要系統的 **PortAudio** 原生函式庫；若缺少，`--list-devices` 會退化用
  `arecord -l` 列出硬體，正式監聽則需先安裝：

  ```bash
  sudo apt install libportaudio2
  ```

- STT 一律經由既有腳本 `~/.hermes/scripts/voice_task/stt_scoped.sh`（cgroup 記憶體隔離，
  避免把 gateway OOM 掉）；**不要**自己載入 faster-whisper。
- `edge-tts` 需要**網路**。
- Discord bot token 放在 `~/.hermes/.env` 的 `DISCORD_BOT_TOKEN`。

> ⚠️ 本機 RAM 吃緊（約 14GB），請勿新增 torch / openai-whisper / pyaudio 等重依賴。

---

## 安裝

專案本身免安裝，直接用模組執行即可：

```bash
git clone <repo> hermes-voice-dispatch
cd hermes-voice-dispatch
PY=~/.hermes/hermes-agent/venv/bin/python3

# 列出麥克風裝置
$PY -m voice_dispatch --list-devices

# 不發網路的端到端演練（推薦第一次先跑這個）
$PY -m voice_dispatch --simulate "幫我把伺服器重啟" --dry-run

# 正式跑（需要麥克風 + 音效輸出 + 網路 + token）
$PY -m voice_dispatch
```

（可選）安裝成 console script `voice-dispatch`：

```bash
$PY -m pip install -e .
```

---

## 設定

預設值即可跑；要調整就複製範例檔再用 `--config` 指定：

```bash
cp config.example.yaml config.yaml
$PY -m voice_dispatch --config config.yaml
```

所有 id / token / 路徑 / 門檻都在設定裡，程式不寫死任何一項。完整鍵值與說明見
[`config.example.yaml`](config.example.yaml)。常用區段：

| 區段 | 重點鍵 | 說明 |
| --- | --- | --- |
| `audio` | `device`, `blocksize`, `players` | 麥克風 index、區塊大小、播放器候選 |
| `clap` | `threshold_mult`, `abs_floor`, `min_gap_sec`, `max_gap_sec` | 拍手偵測門檻（見下方校正） |
| `wake` | `keywords`, `window_sec` | 喚醒詞與喚醒視窗長度 |
| `vad` | `speech_rms_threshold`, `trailing_silence_sec`, `max_record_sec` | 需求錄音的語音偵測 |
| `confirm` | `agree_words`, `disagree_words`, `max_retries` | 回述確認判定 |
| `tts` | `voice`, `ok_prompt`, `confirm_template` | 語音與提示語模板 |
| `stt` | `scoped_script`, `model` | STT 包裝腳本路徑與模型 |
| `discord` | `channel_id`, `guild_id`, `env_file`, `token_env` | Discord 目標與 token 來源 |
| `dispatch` | `hermes_bin`, `prompt_template`, `log_dir` | 派工指令與 prompt |

### Discord token

程式會依序從**行程環境變數** → `discord.env_file`（預設 `~/.hermes/.env`）讀取
`DISCORD_BOT_TOKEN`。dry-run 模式不需要 token。

---

## 拍手門檻校正

拍手偵測用「自適應門檻」：某個音訊區塊的 RMS 若超過「近 `rms_window_sec` 秒 RMS 中位數
× `threshold_mult`」，且在絕對範圍 `[abs_floor, abs_ceil]` 內、呈現尖銳上升緣，就算一次瞬變；
兩次瞬變間隔在 `[min_gap_sec, max_gap_sec]` → 判定雙拍手。

校正步驟：

0. **先確認這顆裝置真的有訊號**（最重要，跳過這步最容易白忙）：

   ```bash
   $PY tools/probe_levels.py            # 量環境底噪
   $PY tools/probe_levels.py --loopback # 從喇叭放 1kHz，驗證麥克風收得到
   ```

   若 `--loopback` 的 max 仍是接近 0，代表**這顆裝置根本是死的**，換一顆
   （`--list` 看有哪些，再 `--device <名稱或index>` 試）。本機就踩過這個坑：
   PipeWire 預設來源 `HiFi__Mic2__source` 是死的（3.5mm 耳麥孔），
   內建麥克風其實是 `HiFi__Mic1__source`。
1. **看背景有多吵**：在你平常的環境放著，觀察 log（`--log-level DEBUG`）。
2. **太難觸發（拍了沒反應）**：
   - 調低 `clap.threshold_mult`（例如 4.0 → 3.0）。
   - 調低 `clap.abs_floor`（例如 0.05 → 0.03），但太低容易被講話/關門誤觸。
   - 確認 `min_gap_sec` / `max_gap_sec` 涵蓋你的拍手節奏（預設 0.12–1.5 秒；
     拍太快可再降 `min_gap_sec`，拍太慢可拉高 `max_gap_sec`）。
3. **太容易誤觸（自己講話或環境音就觸發）**：
   - 調高 `clap.threshold_mult` 或 `clap.abs_floor`。
   - 拍手要「短促、響亮」；持續的大聲（例如音樂）只會產生一個上升緣，不會被當成雙拍手。
4. 每次改完 config 重跑，用 `--once` 搭配 `--log-level DEBUG` 快速驗證。

> 小技巧：`abs_ceil` 是上限，破音級的爆音（RMS 過大）會被排除，避免關門/撞擊誤判成拍手。

---

## systemd（user service）

隨登入自動啟動、崩潰自動重啟：

```bash
mkdir -p ~/.config/systemd/user
cp systemd/voice-dispatch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-dispatch.service

# 看即時 log
journalctl --user -u voice-dispatch.service -f
```

- token 由程式自行從 `~/.hermes/.env` 解析，不需要 `EnvironmentFile`。
- `WorkingDirectory` **必須**是 repo 根目錄（`python -m voice_dispatch` 靠它 import 套件）；
  clone 在別的位置請改掉那一行。
- `Environment=PATH=...` 要含 `hermes` 執行檔所在目錄，否則派工時 spawn 不到 `hermes`。
- 若 venv 或設定檔在別處，請編輯 `ExecStart`（例如加 `--config /path/to/config.yaml`）。
- 想讓服務在關掉登入畫面後仍常駐：`loginctl enable-linger $USER`。

---

## 疑難排解

| 症狀 | 可能原因 / 解法 |
| --- | --- |
| **在跑但拍手永遠沒反應** | 輸入裝置指到收不到聲音的節點。用 `tools/probe_levels.py --loopback` 驗證；本機預設來源是死的 `HiFi__Mic2__source`，要釘 `audio.device: "HiFi__Mic1__source"`。 |
| `PortAudio library not found` | 安裝 `sudo apt install libportaudio2`。`--list-devices` 會自動退化用 `arecord -l`。 |
| `--list-devices` 找不到麥克風 | 用 `arecord -l` 確認硬體；在 config `audio.device` 指定正確 index。 |
| 拍手沒反應 / 一直誤觸 | 見上方「拍手門檻校正」。 |
| 喚醒後聽不到提示音 | 檢查 `ffplay`/`paplay`/`aplay` 是否可用；調整 `audio.players`。 |
| STT 一直失敗 | 確認 `stt.scoped_script` 路徑存在且可執行；看 `/tmp/hermes-stt.log`。 |
| STT 把 gateway OOM | 一定要走 `stt_scoped.sh`，不要自行載入 faster-whisper。 |
| TTS 沒聲音 | `edge-tts` 需要網路；離線時 TTS 會失敗但主流程不會崩潰。 |
| Discord 403 | bot 缺少該頻道的發言 / 建立公開討論串權限。 |
| Discord 401 | token 無效，檢查 `~/.hermes/.env` 的 `DISCORD_BOT_TOKEN`。 |
| Discord 429 | 觸發速率限制，log 會顯示 `retry_after`；稍後再試。 |
| 討論串名稱被截斷 | Discord thread 名上限 100 字元，屬正常行為。 |
| 派工沒動靜 | 看 `dispatch.log_dir` 下的 `dispatch-*.log`；確認 `hermes` 在 PATH。 |

---

## 開發 / 測試

```bash
PY=~/.hermes/hermes-agent/venv/bin/python3
$PY -m pytest -q                # 單元測試（拍手 / VAD / 文字判定 / 設定）
$PY -m compileall voice_dispatch tests
$PY -m voice_dispatch --list-devices
$PY -m voice_dispatch --simulate "測試需求" --dry-run
```

測試/演練時**不會**真的打 Discord、也**不會**真的 spawn `hermes`——請用 `--dry-run` 保護。

### CLI 參數

| 參數 | 說明 |
| --- | --- |
| `--config, -c PATH` | YAML 設定檔（省略則用內建預設值） |
| `--once` | 只跑一輪就結束 |
| `--simulate TEXT` | 跳過麥克風，直接以 TEXT 走 R5/R6 |
| `--dry-run` | 只做本地流程、印出將發送內容，不打 Discord、不 spawn hermes |
| `--list-devices` | 列出麥克風裝置後結束 |
| `--log-level LEVEL` | `DEBUG`/`INFO`/`WARNING`/`ERROR`（預設 INFO） |
| `--log-file PATH` | 覆寫 log 檔路徑 |

---

## 專案結構

```
voice_dispatch/
├── cli.py          # argparse 進入點
├── config.py       # dataclass 設定 + YAML + .env 解析
├── audio.py        # 裝置列舉、InputStream、WAV 讀寫、播放
├── clap.py         # 雙拍手偵測（純狀態機）
├── vad.py          # 能量 VAD 錄音狀態機
├── stt.py          # ffmpeg 正規化 + 呼叫 stt_scoped.sh
├── tts.py          # beep 合成 + edge-tts
├── wake.py         # 拍手迴圈 + 喚醒詞驗證
├── text.py         # 喚醒詞/同意判定、thread 名截斷（純函式）
├── discord_api.py  # Discord REST
├── dispatch.py     # 組 prompt + 背景 spawn hermes
└── daemon.py       # 主狀態機
```

```
tools/
└── probe_levels.py # 麥克風電平探針（校正門檻、驗證裝置真的有訊號）
```
