#!/usr/bin/env bash
# setup_aec.sh — 舊的 AEC 設定方式（**sink 模式**）。⚠️ **已被 monitor.mode 取代，不要用。**
#
# 現行做法（2026-09-28 定案）＝ PipeWire 官方的 `monitor.mode = true`：
#   設定檔 `pipewire/hermes-aec.conf` → 安裝到
#   `~/.config/pipewire/pipewire.conf.d/`，開機自動載入，不需要這支腳本。
#   參考訊號從「預設 sink 的 monitor 埠」取 → **不建 AEC sink、不碰播放路徑**
#   → 喇叭音質完全不受影響。實測衰減 71%，daemon 讀 HermesMicAEC 正常。
#
# 這支腳本走的是 module-echo-cancel 的**預設 sink 模式**：
#   ✅ 衰減更高（96%）
#   ❌ 但必須 `pactl set-default-sink HermesAecSink`，於是**所有**播放都經 webrtc 處理
#      → **喇叭變電音**（使用者原話：「靠，我的喇叭電音」）→ 當場退貨。
#
# 保留原因：(1) 記錄 sink 模式「要三步才算真的開了」這個知識；
#           (2) 誤開時能用 `--revert` 一鍵還原。
#
# sink 模式的三個步驟（缺一就只有二十幾 % 衰減，這是最初誤判「AEC 沒用」的原因）：
#   1. 載入 module-echo-cancel（產生 HermesMicAEC source + HermesAecSink sink）
#   2. **把預設 sink 切成 HermesAecSink** —— 要靠「流經自己 sink 的音訊」當參考訊號
#   3. 把已在播的舊串流 move-sink-input 搬過去
#
# 用法：
#   bash scripts/setup_aec.sh          # sink 模式（**會讓喇叭變電音，不要用**）
#   bash scripts/setup_aec.sh --revert # 還原
set -uo pipefail

MIC_MASTER="${AEC_MIC_MASTER:-alsa_input.usb-Generalplus_WordForum_USB-00.mono-fallback}"
SPK_MASTER="${AEC_SINK_MASTER:-alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink}"
SRC_NAME="HermesMicAEC"
SINK_NAME="HermesAecSink"

have_source() { pactl list sources short | awk '{print $2}' | grep -qx "$1"; }
have_sink()   { pactl list sinks   short | awk '{print $2}' | grep -qx "$1"; }

if [[ "${1:-}" == "--revert" ]]; then
  # 預設 sink 先切回真喇叭，再卸模組（順序相反會讓音訊短暫無出口）
  pactl set-default-sink "$SPK_MASTER" 2>/dev/null || true
  for i in $(pactl list sink-inputs short | awk '{print $1}'); do
    pactl move-sink-input "$i" "$SPK_MASTER" 2>/dev/null || true
  done
  while read -r id name _; do
    [[ "$name" == "module-echo-cancel" ]] && pactl unload-module "$id" 2>/dev/null || true
  done < <(pactl list modules short)
  echo "已還原：預設 sink = $SPK_MASTER，AEC 模組已卸除。"
  echo "記得把 config.yaml 的 audio.device 改回 \"WordForum_USB\"。"
  exit 0
fi

# 1) 載入模組（已存在就跳過，避免每次啟動疊一層）
if have_source "$SRC_NAME" && have_sink "$SINK_NAME"; then
  echo "AEC 已存在，跳過載入。"
else
  pactl load-module module-echo-cancel \
    aec_method=webrtc \
    source_master="$MIC_MASTER" \
    sink_master="$SPK_MASTER" \
    source_name="$SRC_NAME" \
    sink_name="$SINK_NAME" \
    use_master_format=1 \
    aec_args="analog_gain_control=0 digital_gain_control=0 noise_suppression=0 voice_detection=0" \
    >/dev/null || { echo "載入 module-echo-cancel 失敗"; exit 1; }
  echo "已載入 AEC：source=$SRC_NAME sink=$SINK_NAME"
fi

# 2) 預設 sink 一定要是 AEC sink（提供參考訊號）
pactl set-default-sink "$SINK_NAME" || { echo "切換預設 sink 失敗"; exit 1; }

# 3) 把已在播的串流搬到 AEC sink（跳過 AEC 自己的 playback 串流，避免迴圈）
for i in $(pactl list sink-inputs short | awk '{print $1}'); do
  pactl move-sink-input "$i" "$SINK_NAME" 2>/dev/null || true
done

echo "完成。預設 sink = $(pactl get-default-sink)"
echo "驗證衰減：播一段音訊，同時錄 $MIC_MASTER 與 $SRC_NAME 比對 ac_rms（應衰減 >90%）。"
