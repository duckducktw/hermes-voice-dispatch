# CLAUDE.md — hermes-voice-dispatch

**先讀 `SPEC.md`，那是一切的來源。** 本檔只是快速索引。

## 這是什麼
Linux 筆電上的語音派工守護程式：拍手兩下 + 喊「Hermes」喚醒 → 語音說需求 →
回述確認 → 轉發到 Discord #人工智障 並開討論串 → 派工給 Hermes agent 執行並回報。

## 硬規則
- 只用 `~/.hermes/hermes-agent/venv/bin/python3` 這個 venv（numpy / scipy / sounddevice /
  edge-tts 都在裡面）。**不要**新增 torch / openai-whisper / pyaudio 等重依賴。
- STT 一律透過 `~/.hermes/scripts/voice_task/stt_scoped.sh <audio> -`（有 cgroup 保護，
  直接載入 faster-whisper 會 OOM 掉 gateway）。回傳一行 JSON，取 `transcript`。
- 播放音效：`ffplay -nodisp -autoexit -loglevel quiet <file>`（edge-tts 出 mp3）。
- 發 Discord 訊息 / 建 thread：走 REST + `DISCORD_BOT_TOKEN`（在 `~/.hermes/.env`）。
  目標頻道 id 預設 `1392008000197365870`，guild `1340241728959025236`。
- 派工：背景 spawn `hermes -z "<prompt>"`；回報用 `hermes send --to discord:<channel>:<thread>`。
- 任何 id / token / 路徑 / 門檻都放 config，不寫死。

## 常用指令
```bash
PY=~/.hermes/hermes-agent/venv/bin/python3
$PY -m pytest -q                       # 單元測試
$PY -m voice_dispatch --list-devices   # 列麥克風
$PY -m voice_dispatch --simulate "幫我重啟伺服器" --dry-run   # 不發網路的端到端演練
$PY -m voice_dispatch                  # 正式跑（需要有麥克風與音效輸出）
```

## 別踩的坑
- 本機 RAM 吃緊（14GB），STT 一定要走 stt_scoped.sh。
- Discord thread 名上限 100 字元；`auto_archive_duration` 用 1440（分鐘）。
- `edge-tts` 需要網路。
- 測試時**不要**真的打 Discord API、不要真的 spawn hermes；用 `--dry-run`。
