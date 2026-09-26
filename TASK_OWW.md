# TASK — 把喚醒詞從「自製 Vosk 串接」改成業界正規做法（openWakeWord）

## 背景（已由 Hermes 實測，勿再重驗）
本專案目前的喚醒偵測是**自製的假 KWS**：`voice_dispatch/cascade.py` 用
「Vosk 限制詞彙當 hey 閘門（可用 partial 觸發）→ 全詞彙解碼 2s 窗內找 hey hermes bigram」。
實測（7 天 journal）證明這個做法 specificity 極差：

- `hey 閘門觸發但窗內未見 hermes` = **1688 次**；真正喚醒只有 **18 次** → 命中率 **1%**。
- 每個閘門觸發都跑一次全詞彙 Vosk 解碼 → 常駐 CPU 4.3% 單核空轉。
- 閘門為雜音誤觸發（轉錄 `yeah`/`the`/`okay`/`hello`…），且出現 2s/22s 固定節奏
  （疑似 `reset()` 未清 gate recognizer 造成的自我再觸發）。

**原因是架構錯**：用「通用 ASR + 限制詞彙」冒充 KWS。業界標準做法是用**訓練過的
神經 KWS 模型**（openWakeWord / microWakeWord / Porcupine），並搭配：
`VAD 閘控` + `confirmation_frames（連續 N 幀超標才算）` + `sensitivity 門檻`。

## 關鍵事實
- **`hey_hermes` 的訓練好模型已經在本機**：
  `/home/user/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx`（另有 `.tflite`）。
  這是 Hermes 內建 wake word 功能附的官方模型，**零訓練成本**。
- `openwakeword` 已經裝在專案用的 venv（`~/.hermes/hermes-agent/venv`），**不需新增依賴**。
- 因此 SPEC R1 第 3 點「openwakeword 不支援自訂詞」**已過時**，本任務要推翻它。

## 目標（最小改動，可量測）
新增一個 **openWakeWord 喚醒引擎**，與現有 Vosk 串接並存可切換，並用**同一個語料**
做出 FP/FN/延遲的頭對頭比較，證明哪個好。**不要動**錄音 / STT / 派工 / TTS / relay 流程。

### 1. 新模組 `voice_dispatch/oww.py`
- `class OwwSpotter`：介面與 cascade/daemon 現行用法一致：
  - `__init__(self, model_path: str, threshold: float, confirmation_frames: int,
      vad_threshold: float = 0.0, logger=None)`
  - `feed(self, block: np.ndarray) -> bool`：吃 16kHz / mono / int16、blocksize=1024 的 block，
    內部緩衝成 openWakeWord 要的 80ms（1280 samples）幀再餵。
    **連續 `confirmation_frames` 幀分數 ≥ `threshold`** 才回 True（之後自行 reset/冷卻）。
  - `reset(self)`：回到監聽狀態。
  - 用 `openwakeword.model.Model(wakeword_models=[model_path], inference_framework="onnx")`；
    若 `vad_threshold > 0` 就開 openWakeWord 內建 Silero VAD 閘控。
  - 載入失敗 / 模型不存在 → 記 warning 並讓 daemon 能優雅退回 `kws`，不可 crash。

### 2. 接進 daemon（`wake.mode`）
- `config.py` 的 `WakeConfig` 新增：
  - `mode` 允許值多一個 `"openwakeword"`（**預設仍維持 `"kws"`，不要改預設**）。
  - `oww_model: str = "~/.hermes/hermes-agent/tools/wakewords/hey_hermes.onnx"`
  - `oww_threshold: float = 0.6`      # 對應官方 sensitivity
  - `oww_confirmation_frames: int = 3`
  - `oww_vad_threshold: float = 0.0`  # 0 = 關
- `daemon.py`：`mode == "openwakeword"` 時走新的 `_wait_for_wake_oww()`，
  結構比照 `_wait_for_wake_kws()`：**同一條 input_stream**、
  沿用 `_echo_muted()` 半雙工、`frozen` 偵測、`_recover_audio()`。
  命中時 log `喚醒詞確認：hey hermes（openWakeWord score=%.2f）` 並回 True。
- 保留 `_wait_for_wake_kws()` 完全不動。

### 3. 擴充評測 `tools/eval_wake.py`
- 新增 `--engine {vosk,oww,cascade}`（保留現有 cascade 行為）+ `--oww-model`、
  `--oww-threshold`、`--oww-confirmation-frames`、`--oww-threshold-sweep a,b,c`。
- 覺醒定義與現行一致：**整個檔案跑完任一時點命中**。
- 輸出沿用現有混淆矩陣 + 逐句誤判清單，並多印 **平均喚醒延遲（ms）**。

### 4. 跑比較並貼真實輸出
語料 `/tmp/wake_corpus`（664 個 wav，已存在；若缺就 `tools/build_wake_corpus.py` 重建）。
必須實跑並在結論貼出：
```
# baseline
python3 tools/eval_wake.py --corpus /tmp/wake_corpus --cascade \
    --stage1-model ~/.local/share/hermes-voice-dispatch/vosk-model-small-en-us-0.15 \
    --stage2-model <stage2 model if present>
# 候選
python3 tools/eval_wake.py --corpus /tmp/wake_corpus --engine oww \
    --oww-threshold-sweep 0.3,0.5,0.6,0.7,0.8
```
做成 FP / FN / 延遲對照表。

### 5. 測試
- 新 `tests/test_oww.py`：載入模型、餵一段含 hey hermes 的合成音（可用 edge-tts 或語料）→ 應命中；
  餵靜音/雜音 → 不命中。模型不可用時 `pytest.skip`。
- `~/.hermes/hermes-agent/venv/bin/python3 -m pytest -q` **全綠**（現有測試不可壞）。

## 硬規則（違反即失敗）
- 只用 `~/.hermes/hermes-agent/venv/bin/python3`。**不新增** torch / sherpa / pyaudio 等重依賴。
- **不要** git commit / push。**不要** systemctl restart 任何服務（改完交給人重啟）。
- **不要**打 Discord API、**不要**真的 spawn `hermes`。測試一律離線 / dry-run。
- 不改預設 `wake.mode`（維持 `kws`）；不動 STT / 派工 / TTS / relay。
- 讀 SPEC.md 與 CLAUDE.md；任何 id/token/路徑/門檻一律放 config 不寫死。

## 交付（你自己的回報必須含）
1. 改了哪些檔案（逐檔一句話）。
2. FP/FN/延遲對照表（真實終端輸出，非編造）。
3. `pytest -q` 的尾巴輸出。
4. 一句結論：openWakeWord 是否勝過現行 Vosk 串接、建議的 `oww_threshold` 為何。
5. 若有做不到的部分，直說卡在哪，不要假造數據。
