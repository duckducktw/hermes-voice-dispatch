# hermes-voice-dispatch

Linux 筆電上的**語音派工守護程式**。

> 喊「**hey Hermes**」→ 提示音（懂咚）→ 說出需求 → 提示音（咚懂）→
> 需求當成「一則 @Hermes 的使用者訊息」發進 Discord `#人工智障` →
> gateway 走它原本那條路處理（同一條 session、逐字串流、可直接追問）→
> 結果**完整唸出來**。

實作細節與逐條需求見 [`SPEC.md`](SPEC.md)；給未來 agent 的快速索引見 [`CLAUDE.md`](CLAUDE.md)。

---

## 運作流程

```
idle ──「hey Hermes」(openWakeWord)──▶ 懂咚 ──▶ 錄需求(Silero VAD) ──▶ 咚懂
  ▲                                                                    │
  │                                                                    ▼
  └──── 語音完整唸出結果 ◀──── gateway 處理 ◀──── webhook relay 進 Discord
```

- **R1 喚醒**：`openWakeWord` 神經網路喚醒詞（`hey_hermes.onnx`），常開只做推論，
  **不再需要拍手**。判定是**雙層**：強命中＝單幀 ≥ `wake.oww_threshold`；
  弱命中＝`oww_window_frames` 幀內有 ≥ `oww_relaxed_hits` 幀超過 `oww_relaxed_threshold`。
- **R2 提示音**：只用兩顆 chime、**不講話**（`tts.prompt_mode="chime"`）。
  **懂咚＝收到喚醒**、**咚懂＝錄音結束／沒收到錄音**，其他時間不出聲。
- **R3 轉錄**：Silero VAD 收音 → 正規化 16k mono wav → 經 `stt_scoped.sh` 轉錄。
- **R4 複述**：只是「告知」，**不等待確認**（2026-09-25 起沒有獨立 confirm 階段）。
- **R5 轉發**：`relay` 路徑用 webhook 在母頻道發 **1 則**「<@Hermes> 需求原文」，
  模仿 bot 的名字與頭像；gateway 的 `force_thread_channels` 自己開串接手。
- **R6 回報**：gateway 在串內逐字串流回覆，daemon 把結果**完整唸完不截斷**。

> 舊做法（拍手兩下 → 錄 2.5s → 對那段做 STT 比對字串）仍留在 `clap.py` / `wake.py`
> 當退路（`wake.mode="clap"`），但**預設不用**：實測慢（命中後還要 5s 才回應）、
> 且短音訊 STT 極易吐幻覺。另有 `cascade.py`（Vosk「hey」閘門 → 全詞彙確認）備選。

---

## 需求與依賴

**一律使用 Hermes venv 的 python**，依賴都已裝在裡面：

```bash
PY=~/.hermes/hermes-agent/venv/bin/python3
```

- Python 3.11、`numpy` / `scipy` / `sounddevice` / `onnxruntime` / `edge-tts` / `PyYAML`
- 系統工具：`ffmpeg`、`ffprobe`、`ffplay`（或 `paplay` / `aplay`）、`arecord`
- `sounddevice` 需要系統的 **PortAudio**：`sudo apt install libportaudio2`
- STT 一律經 `~/.hermes/scripts/voice_task/stt_scoped.sh`（cgroup 記憶體隔離，
  避免把 gateway OOM 掉）；**不要**自己載入 faster-whisper。
- TTS 需要**網路**（Gemini TTS / edge-tts）。
- Discord token 放 `~/.hermes/.env` 的 `DISCORD_BOT_TOKEN`；
  relay 的 webhook URL 放 `DISCORD_RELAY_WEBHOOK_URL`。

> ⚠️ 本機 RAM 吃緊（約 14GB），請勿新增 torch / openai-whisper / pyaudio 等重依賴。

---

## 執行

```bash
PY=~/.hermes/hermes-agent/venv/bin/python3

$PY -m voice_dispatch --list-devices                       # 列麥克風
$PY -m voice_dispatch --check-audio                        # 檢查輸入裝置真的有訊號
$PY -m voice_dispatch --simulate "幫我重啟伺服器" --dry-run  # 不打網路的端到端演練
$PY -m voice_dispatch --config config.yaml                 # 正式跑
```

（可選）安裝成 console script：`$PY -m pip install -e .`

---

## 設定

所有 id / token / 路徑 / 門檻都在設定裡，程式不寫死任何一項。
完整鍵值見 [`config.example.yaml`](config.example.yaml)。現行區段：

| 區段 | 重點鍵 | 說明 |
| --- | --- | --- |
| `audio` | `device`, `blocksize`, `players` | 麥克風、區塊大小、播放器候選 |
| `wake` | `mode`, `oww_model`, `oww_threshold`, `oww_relaxed_*` | 喚醒引擎與雙層門檻 |
| `tts` | `engine`, `voice`, `gemini_style`, `speak_cps`, `speak_chunk_chars` | 語音合成與語速 |
| `discord` | `channel_id`, `guild_id`, `user_id`, `card_template` | Discord 目標與訊息模板 |
| `relay` | `enabled` | webhook relay（當成使用者訊息）；停用則回退 spawn `hermes -z` |
| `confirm` | `retry_on_no_speech` | 沒收到錄音是否重問（預設 false） |
| `webhook` | `enabled` | 舊的本機 webhook route，**已停用** |

### TTS（Gemini，預設）

`tts.engine="gemini"` 比 edge-tts 更像真人，並能用「風格指示」演出商業大佬口吻。
注意事項：

- 風格指示**必須**寫成分節標題（`# 風格指示` / `# 台詞`），否則 Gemini 會把指示也唸出來。
- 語速**不要**靠 prompt 文字控（實測不同措辭輸出逐位元相同）。用 `speak_cps`
  （每秒幾個字）正規化，跨 model 一致；`speed` 只是備援固定倍率。
- 免費層**每個 model 每天 10 次**配額 → 依序輪替 `gemini_model` /
  `gemini_model_fallbacks`，全失敗才退回 edge-tts。
- 長文（> `speak_chunk_chars`，預設 240 字）會分段合成再接起來，**完整唸完不截斷**。

---

## systemd（user service）

```bash
mkdir -p ~/.config/systemd/user
cp systemd/voice-dispatch.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-dispatch.service
journalctl --user -u voice-dispatch.service -f
```

- token 由程式自行從 `~/.hermes/.env` 解析，不需要 `EnvironmentFile`。
- `WorkingDirectory` **必須**是 repo 根目錄（`python -m voice_dispatch` 靠它 import）。
- `Environment=PATH=...` 要含 `hermes` 執行檔所在目錄，否則回退路徑 spawn 不到。
- 想在關掉登入畫面後常駐：`loginctl enable-linger $USER`。

---

## 疑難排解

| 症狀 | 可能原因 / 解法 |
| --- | --- |
| **喚醒叫不響／太鬆** | 看 log 的 `openWakeWord 觀測：近 5 秒最高分 X…`。<br>(a) X 衝得上去但沒醒 → 降 `oww_relaxed_threshold` 或 `oww_relaxed_hits`。<br>(b) 太鬆 → 升 `oww_threshold`。<br>(c) X < 0.05 → 音訊沒進模型（裝置／音量／取樣率）。<br>⚠️ **不要用「連續 N 幀」條件**：分數是單幀尖峰（實測 0.938 只維持 1 幀）。 |
| **語音唸到一半突然超大聲「沙」** | Gemini 回傳格式判錯，把壓縮位元組當裸 PCM 解 → 全振幅白噪音。已於 `_gemini_write_audio` 修正（容器嗅探 + mimeType 取樣率 + `_is_noise_burst` 守門，偵測到就換 model 重合成）。log 會印 `偵測到爆音／雜訊：max=… crest=…`。 |
| **語音回報只唸一半就停** | (1) `tts.speak_result_max_chars` 為正數 → 只唸前 N 字，**0＝完整唸完**（預設）。<br>(2) Gemini 單次輸出有長度上限，長文靠 `speak_chunk_chars` 分段；另用「字/秒 ≫ 目標」自動偵測截斷並切半重合成。診斷：`tools/probe_speak_pipeline.py 1500`。 |
| 在跑但完全沒反應 | 輸入裝置指到收不到聲音的節點。用 `--check-audio` 或 `tools/probe_levels.py --loopback` 驗證。本機踩過：PipeWire 的 `HiFi__Mic2__source`（3.5mm 孔）是死的。 |
| `PortAudio library not found` | `sudo apt install libportaudio2`。 |
| STT 一直失敗 | 確認 `stt.scoped_script` 存在且可執行；看 `/tmp/hermes-stt.log`。 |
| STT 把 gateway OOM | 一定要走 `stt_scoped.sh`，不要自行載入 faster-whisper。 |
| TTS 沒聲音 | 需要網路；離線時 TTS 失敗但主流程不崩潰。 |
| Discord 401 / 403 / 429 | token 無效 / bot 缺發言或建串權限 / 速率限制（log 有 `retry_after`）。 |
| relay 沒進 gateway | gateway 的 `.env` 要有 `DISCORD_ALLOW_BOTS=mentions`，且 `streaming.enabled=true`；`DISCORD_RELAY_WEBHOOK_URL` 要填。 |
| 討論串名稱被截斷 | Discord thread 名上限 100 字元，屬正常行為。 |

---

## 開發 / 測試

```bash
PY=~/.hermes/hermes-agent/venv/bin/python3
$PY -m pytest -q                                   # 全部單元測試
$PY -m compileall voice_dispatch tests
$PY -m voice_dispatch --simulate "測試需求" --dry-run
```

測試/演練時**不會**真的打 Discord、也**不會**真的 spawn `hermes`——請用 `--dry-run` 保護。

### CLI 參數

| 參數 | 說明 |
| --- | --- |
| `--config, -c PATH` | YAML 設定檔（省略則用內建預設值） |
| `--once` | 只跑一輪就結束 |
| `--simulate TEXT` | 跳過麥克風，直接以 TEXT 走後段流程 |
| `--dry-run` | 只做本地流程，不打 Discord、不 spawn hermes |
| `--list-devices` | 列出麥克風裝置後結束 |
| `--check-audio` | 檢查輸入裝置是否真的有訊號後結束 |
| `--log-level LEVEL` | `DEBUG`/`INFO`/`WARNING`/`ERROR`（預設 INFO） |
| `--log-file PATH` | 覆寫 log 檔路徑 |

---

## 專案結構

```
voice_dispatch/
├── cli.py          # argparse 進入點
├── config.py       # dataclass 設定 + YAML + .env 解析
├── audio.py        # 裝置列舉、InputStream、WAV 讀寫、播放
├── oww.py          # openWakeWord 喚醒（現行預設，雙層門檻）
├── kws.py          # 神經網路喚醒詞 / Silero VAD 封裝
├── cascade.py      # 串接式喚醒：Vosk「hey」閘門 → 全詞彙確認（備選）
├── clap.py         # 雙拍手偵測（舊做法，留作退路）
├── wake.py         # 喚醒迴圈（含舊的拍手 + STT 驗證路徑）
├── vad.py          # 能量 VAD 錄音狀態機
├── stt.py          # ffmpeg 正規化 + 呼叫 stt_scoped.sh
├── tts.py          # chime 合成 + Gemini/edge TTS + 音訊解碼守門
├── text.py         # 喚醒詞/同意判定、thread 名截斷（純函式）
├── discord_api.py  # Discord REST + webhook relay
├── dispatch.py     # 組 prompt + 背景 spawn hermes（回退路徑）
└── daemon.py       # 主狀態機

tools/
├── probe_levels.py         # 麥克風電平探針（驗證裝置真的有訊號）
├── make_cues.py            # 重新產生提示音素材（懂咚／咚懂）
├── probe_speak_pipeline.py # 長文 TTS 分段/截斷診斷
├── probe_long_tts.py       # Gemini 單次輸出長度上限實測
├── probe_oww_files.py      # 對音檔跑 openWakeWord 打分
├── probe_live_wake.py      # 即時喚醒觀測
├── eval_wake.py            # 喚醒詞離線評估
├── build_wake_corpus.py    # 產生喚醒詞評估語料
├── wake_matrix.py          # 門檻掃描矩陣
├── probe_api_server.py     # Hermes api_server 探針
└── probe_guard.py          # 派工守門探針

assets/chime/       # 提示音素材（低沉＋快＋抖動，全合成、無版權問題）
systemd/            # user service 範例
```
