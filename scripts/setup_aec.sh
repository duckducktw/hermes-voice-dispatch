#!/usr/bin/env bash
# setup_aec.sh — 建立 AEC（回音消除）麥克風。
#
# ⚠️⚠️ 2026-09-28 **此方案已被使用者退貨，預設不啟用，不要自動跑這支腳本** ⚠️⚠️
#
# 當初動機（使用者：「電腦發出來的聲音不納入語音辨識，電腦發出來不被錄」）與實測結果：
#   ✅ 回音消除本身有效：同一段 TTS，原始 mic ac_rms 0.0763 → AEC 0.00303 ＝ **衰減 96%**
#      （第一次只量到 27% 是因為沒做第 2 步，見下方）。
#   ❌ **但喇叭輸出會變電音** —— 使用者原話：「靠，我的喇叭電音」。原因：把預設 sink 切成
#      AEC sink 之後，**所有** 播放音訊都要經過 webrtc AEC 處理，這台機器上
#      （USB mic 48k mono ↔ Speaker sink 48k stereo，需重取樣/聲道轉換）音質被毀。
#   ❌ 而且 daemon 讀 AEC source 後 journal 完全停住（80 秒沒有任何「觀測」輸出）。
#
# → 結論：**在這台機器上不要用 AEC**。想避免電腦聲音造成誤喚醒，改用
#   `wake.mute_while_system_audio`（讀 sink monitor 電平的來源閘門，預設 false，
#   細節見 config.yaml）——那個不會碰音訊路徑，所以不會影響音質。
#
# 這支腳本保留下來只為兩件事：(1) 記錄「怎麼做才算真的開了 AEC」；
# (2) 萬一哪天誤開了，能用 `--revert` 一鍵還原。
#
# 若真要重測（例如換了音效裝置），三個步驟缺一不可：
#   1. 載入 module-echo-cancel（產生 HermesMicAEC source + HermesAecSink sink）
#   2. **把預設 sink 切成 HermesAecSink** —— AEC 要靠「流經自己 sink 的音訊」當參考訊號，
#      播放不走它就沒東西可消，衰減只會有二十幾 %（這是第一次誤判「AEC 沒用」的原因）。
#   3. 把已經在播的舊串流 move-sink-input 搬過去（否則它們仍走舊 sink）。
#   然後 config.yaml 的 audio.device 要改成 "HermesMicAEC"。
#
# 用法：
#   bash scripts/setup_aec.sh          # 建立/確保啟用（**會讓喇叭變電音，慎用**）
#   bash scripts/setup_aec.sh --revert # 還原（卸掉模組、預設 sink 切回喇叭）
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
