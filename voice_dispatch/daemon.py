"""主狀態機：idle → wake → record → forward → dispatch。

（2026-09-25 起沒有獨立的 confirm 階段：複述只是告知、不等待回覆。）

- 正式模式：持續監聽麥克風，雙拍手 + 喚醒詞喚醒後走完整流程。
- --once：只跑一輪。
- --simulate "<文字>"：跳過麥克風，直接以該文字走 R3 之後流程。
- --dry-run：只做本地流程、印出「將發送/派工的內容」，不打 Discord、不 spawn hermes。
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np

from . import audio, cascade, dispatch, kws, oww, stt, tts, wake
from .config import Config
from .discord_api import DiscordClient, user_avatar_url
from .text import classify_confirmation, make_thread_name, normalize, spoken_summary
from .vad import VadSegmenter, VadState


def _as_list(x) -> list:
    """把 Discord GET 的回應正規化成 list（出錯時可能是 dict 或 None）。"""
    return x if isinstance(x, list) else []


def pick_report(rows, last_spoken_id: int) -> Optional[tuple]:
    """從「已結束回合」的 assistant 訊息中挑出該唸的那一則。

    `rows`＝[(id, role, content, finish_reason), ...]（同一 session、依 id 排序，可由
    SQL 先篩掉非 (assistant, finish_reason='stop') 的列）。回傳 `(id, content)`＝最新
    一則有內容、且 `id > last_spoken_id` 的訊息；沒有就回 `None`（＝這一輪沒有新結果）。

    2026-09-26 (c) 方案：重新武裝時靠 `last_spoken_id` 記住上次唸到哪一則，
    所以同一則不會被唸第二次。
    """
    best = None
    for mid, role, content, fr in rows:
        if role != "assistant" or fr != "stop":
            continue
        if not str(content or "").strip():
            continue
        if int(mid) <= int(last_spoken_id):
            continue
        best = (int(mid), str(content))
    return best

log = logging.getLogger("voice_dispatch")


class VoiceDispatcher:
    def __init__(self, cfg: Config, *, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self._stop = False
        self._last_recover = 0.0
        self._recover_count = 0
        self._healthy_since = 0.0   # 連續健康起算點（見 _log_stream_health）
        self._silero = None         # 惰性建立；False = 載入失敗，退回 RMS 門檻

    # ------------------------------------------------------------------
    # 生命週期
    # ------------------------------------------------------------------
    def request_stop(self, *_args) -> None:
        log.info("收到停止訊號，準備乾淨關閉…")
        self._stop = True

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

    # ------------------------------------------------------------------
    # 麥克風擷取小工具
    # ------------------------------------------------------------------
    def _collect_seconds(self, stream, seconds: float) -> np.ndarray:
        """從已開啟的 stream 收集約 seconds 秒的樣本。"""
        blocksize = self.cfg.audio.blocksize
        samplerate = self.cfg.audio.samplerate
        needed = int(round(seconds * samplerate))
        chunks: List[np.ndarray] = []
        got = 0
        for block in audio.read_blocks(stream, blocksize):
            chunks.append(block)
            got += len(block)
            if got >= needed or self._stop:
                break
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)[:needed]

    def _get_silero(self):
        """惰性建立 Silero VAD；載入失敗就退回 RMS 門檻（回傳 None）。"""
        if self._silero is None:
            try:
                self._silero = kws.SileroVad(
                    self.cfg.vad.silero_threshold,
                    model_path=getattr(self.cfg.vad, "silero_model", ""),
                    rms_gate=getattr(self.cfg.vad, "silero_rms_gate", 0.0),
                    logger=log,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Silero VAD 載入失敗（%s）→ 改用 RMS 門檻", exc)
                self._silero = False
        if self._silero is False:
            return None
        return self._silero

    def _drain(self, stream, seconds: float) -> None:
        """丟掉串流裡積壓的音訊（含播放 TTS 期間累積的）。

        不做的話，開始錄需求時會先讀到我們自己剛剛播的提示語，
        VAD 會把它當成使用者需求 → STT → 派工一個假任務（2026-09-25 實際踩到）。
        """
        need = int(max(0.0, seconds) * self.cfg.audio.samplerate)
        bs = self.cfg.audio.blocksize
        got = 0
        while got < need and not self._stop:
            data, _ = stream.read(bs)
            got += int(np.asarray(data).size)

    def _record_utterance(self, stream) -> Optional[np.ndarray]:
        """能量 VAD 錄一段話。回傳樣本；前置靜音逾時回傳 None。

        含 pre-roll：語音起點前多保留 `preroll_keep_sec` 秒的滾動緩衝，
        避免「VAD 判定得比人開口晚」把字頭切掉——手機 mic 隔著桌面收音時，
        起頭那幾個字常常過不了門檻，切掉字頭就很容易被 STT 轉成幻覺。
        """
        seg = VadSegmenter(self.cfg.vad)
        blocksize = self.cfg.audio.blocksize
        samplerate = self.cfg.audio.samplerate
        # 先排掉播放提示語期間積在緩衝裡的自家人聲（否則會錄到自己剛講的話）
        self._drain(stream, self.cfg.vad.settle_sec)
        if self._stop:
            return None
        block_dur = blocksize / float(samplerate)
        keep_n = max(1, int(round(self.cfg.vad.preroll_keep_sec / block_dur)))
        pre: deque = deque(maxlen=keep_n)      # 語音開始前的滾動緩衝
        chunks: List[np.ndarray] = []
        silero = self._get_silero()
        idx = 0
        for block in audio.read_blocks(stream, blocksize):
            if self._stop:
                break
            t = idx * block_dur
            idx += 1
            rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            # Silero 以 80ms 為單位推論，答案會比當前區塊晚一點（可接受）
            voiced = silero.feed(block) if silero is not None else None
            was_started = seg.started
            state = seg.feed(rms, t, voiced=voiced)
            if seg.started and not was_started:
                chunks.extend(pre)             # 補回字頭
                pre.clear()
            if seg.started:
                chunks.append(block)
            else:
                pre.append(block)
            if state == VadState.DONE:
                break
            if state == VadState.TIMEOUT:
                return None
        if not chunks:
            return None
        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    # R1：喚醒
    # ------------------------------------------------------------------
    def _echo_muted(self) -> bool:
        """半雙工閘：我們自己在出聲（或剛出聲完）時，不做喚醒偵測。

        兩個來源都要擋：
          ① 正在播（`audio.output_busy()`）——可能是背景 thread 在播語音回報；
          ② 剛播完的殘響／裝置緩衝（`wake.echo_guard_sec` 秒內）。

        不做的話它會把自己的提示音／TTS 收回來當成「hey hermes」→ 誤喚醒 →
        再播一輪提示音 → 連響（2026-09-26 使用者回報的症狀）。
        """
        if audio.output_busy():
            return True
        guard = float(getattr(self.cfg.wake, "echo_guard_sec", 0.0) or 0.0)
        if guard <= 0.0:
            return False
        return audio.output_quiet_sec() < guard

    def _skip_echo_blocks(self, blocks):
        """把「我們自己在出聲」期間的區塊濾掉（給不自己判狀態的消費端用）。"""
        muted = False
        for block in blocks:
            if self._stop:
                return
            if self._echo_muted():
                if not muted:
                    muted = True
                    log.debug("自己在出聲 → 暫停喚醒偵測（半雙工）")
                continue
            if muted:
                muted = False
                log.debug("恢復喚醒偵測")
            yield block

    def _wait_for_wake_kws(self) -> bool:
        """串接式 KWS 喚醒，命中就回 True。

          第一階段（低功耗，常開）：只認「hey」的限制詞彙閘門，可用 partial 即時觸發。
          第二階段（高精度，僅對候選）：觸發後對一小段音訊做**全詞彙**解碼，確認真的
          講了「hey hermes」才喚醒——**一確認就立刻醒**，不等句子結束的靜音判定。
        邏輯都封裝在 `cascade.WakeCascade`（與 I/O 無關，可離線評測）。
        """
        wake_detect = cascade.WakeCascade(self.cfg, logger=log)
        while not self._stop:
            frozen = {"hit": False, "blocks": 0}
            with audio.input_stream(self.cfg.audio) as stream:
                self._log_stream_health(stream)
                if self._stop:
                    return False
                blocks = audio.read_blocks_watched(
                    stream, self.cfg.audio.blocksize,
                    frozen_max_blocks=self.cfg.audio.frozen_max_blocks,
                    on_frozen=self._frozen_cb(frozen),
                )
                muted = False
                for block in blocks:
                    if self._stop:
                        return False
                    if self._echo_muted():
                        if not muted:
                            muted = True
                            log.debug("自己在出聲 → 暫停喚醒偵測（半雙工）")
                        continue
                    if muted:
                        muted = False
                        # 靜音期間完全沒餵過偵測器，但 ring 裡還留著播放前的音訊
                        wake_detect.reset()
                        log.debug("恢復喚醒偵測")
                    hit = wake_detect.feed(block)
                    if hit:
                        log.info("喚醒詞確認：%s", hit)
                        return True
                if frozen["hit"]:
                    log.warning("擷取串流凍結（連續 %d 個區塊位元完全相同）→ mic 掛了",
                                frozen["blocks"])
                    self._recover_audio()
        return False

    def _wait_for_wake_oww(self) -> bool:
        """用自訂 hey_hermes ONNX 模型監聽喚醒詞。

        與既有 KWS 路徑相同：只持有一條 input stream、沿用半雙工靜音、串流凍結
        偵測與恢復。模型/執行期不可用時，安全回退到既有的 KWS 路徑。
        """
        try:
            wake_detect = oww.OwwSpotter(
                self.cfg.wake.oww_model,
                self.cfg.wake.oww_threshold,
                self.cfg.wake.oww_confirmation_frames,
                self.cfg.wake.oww_vad_threshold,
                logger=log,
            )
        except oww.OwwUnavailable:
            log.warning("openWakeWord 不可用，回退 wake.mode=kws 的既有串接引擎。")
            return self._wait_for_wake_kws()

        while not self._stop:
            frozen = {"hit": False, "blocks": 0}
            with audio.input_stream(self.cfg.audio) as stream:
                self._log_stream_health(stream)
                if self._stop:
                    return False
                blocks = audio.read_blocks_watched(
                    stream, self.cfg.audio.blocksize,
                    frozen_max_blocks=self.cfg.audio.frozen_max_blocks,
                    on_frozen=self._frozen_cb(frozen),
                )
                muted = False
                for block in blocks:
                    if self._stop:
                        return False
                    if self._echo_muted():
                        if not muted:
                            muted = True
                            log.debug("自己在出聲 → 暫停喚醒偵測（半雙工）")
                        continue
                    if muted:
                        muted = False
                        wake_detect.reset()
                        log.debug("恢復喚醒偵測")
                    if wake_detect.feed(block):
                        log.info(
                            "喚醒詞確認：hey hermes（openWakeWord score=%.2f）",
                            wake_detect.latest_score,
                        )
                        return True
                if frozen["hit"]:
                    log.warning("擷取串流凍結（連續 %d 個區塊位元完全相同）→ mic 掛了",
                                frozen["blocks"])
                    self._recover_audio()
        return False

    def wait_for_wake(self) -> bool:
        """監聽喚醒詞；喚醒成功回傳 True。

        依 `cfg.wake.mode` 選引擎：
          - "kws"（預設）：**串接式關鍵詞偵測** = 第一階段「hey」閘門（Vosk 限制
            詞彙、常開、可用 partial 即時觸發）→ 第二階段「hermes」確認（全詞彙
            解碼 + bigram 規則）。不需要拍手、不對喚醒詞做 STT。實測 664 句近似
            發音語料誤判 0/600、漏判 0/64，最大喚醒延遲 ~110ms。
          - "openwakeword"：以本機已訓練的 hey_hermes ONNX 模型直接偵測；模型
            不可用時會記 warning 並安全回退到既有 "kws"。
          - "clap"：舊做法（拍手兩下 → 錄一段 → STT 比對字串），只留作退路。
        """
        if self.cfg.wake.mode == "kws":
            return self._wait_for_wake_kws()
        if self.cfg.wake.mode == "openwakeword":
            return self._wait_for_wake_oww()
        return self._wait_for_wake_clap()

    def _wait_for_wake_clap(self) -> bool:
        """開麥克風監聽雙拍手 + 喚醒詞。喚醒成功回傳 True。

        重點：**只開一條串流並長期持有**，健檢與凍結偵測都跑在同一條串流上。
        本機 DMIC 的怪癖是「多開／重開常常只拿到凍結的直流」，所以另外開一條
        去做健檢會把唯一正常的開檔搶走，反而讓監聽中的那條變成死的。
        """
        while not self._stop:
            frozen = {"hit": False, "blocks": 0}
            recovered = False
            with audio.input_stream(self.cfg.audio) as stream:
                self._log_stream_health(stream)
                if self._stop:
                    return False
                blocks = audio.read_blocks_watched(
                    stream,
                    self.cfg.audio.blocksize,
                    frozen_max_blocks=self.cfg.audio.frozen_max_blocks,
                    on_frozen=self._frozen_cb(frozen),
                )
                detected = wake.run_clap_loop(
                    self._skip_echo_blocks(blocks), self.cfg,
                    should_stop=lambda: self._stop,
                )
                if self._stop:
                    return False
                if frozen["hit"]:
                    log.warning(
                        "擷取串流凍結（連續 %d 個區塊位元完全相同）→ mic 掛了",
                        frozen["blocks"],
                    )
                    recovered = True
                elif not detected:
                    log.info("監聽串流結束（裝置被搶走？）→ 重新開啟。")
                else:
                    log.info("偵測到雙拍手，錄喚醒詞視窗…")
                    samples = self._collect_seconds(stream, self.cfg.wake.window_sec)
                    try:
                        # 喚醒詞用快模型（small，約 2.6s）而不是 large-v3（7s）：
                        # 這裡只需要比對「hermes」一個詞，不值得花 7 秒。
                        transcript = stt.transcribe_samples(
                            samples, self.cfg, logger=log,
                            model=self.cfg.stt.wake_model,
                        )
                    except stt.SttError as exc:
                        log.warning("喚醒詞 STT 失敗：%s", exc)
                        self._cooldown()
                        continue
                    log.info("喚醒詞視窗轉錄：%r", transcript)
                    if wake.verify_wake_word(transcript, self.cfg):
                        return True
                    log.info("未命中喚醒詞，靜默重置。")
                    # 靜默重置：不發任何音效/訊息，繼續監聽
                    self._cooldown()
            if recovered:
                self._recover_audio()
        return False

    # ------------------------------------------------------------------
    # 麥克風健檢 / 凍結自動恢復
    # ------------------------------------------------------------------
    def _frozen_cb(self, state: dict):
        """產生給 read_blocks_watched 的回呼，把凍結事件記進 state。"""

        def cb(count: int) -> None:
            state["hit"] = True
            state["blocks"] = count

        return cb

    def _log_stream_health(self, stream, attempts: int = 6) -> None:
        """在「同一條」串流上量 crest，把「死訊號」講清楚。

        不要為了健檢另外開 stream：本機多開會拿到凍結訊號，反而害到監聽中的
        那條（2026-09-25 實際踩到的坑）。

        手機麥克風（PhoneMic）是 on-demand 的：bridge 要約 1 秒才會把串流接上，
        開機瞬間量到的是數位靜音。所以像死訊號時要「等一下再重試」，不要第一次
        就判死（2026-09-25 實測：每次重啟都誤報死訊號）。
        """
        try:
            bs = self.cfg.audio.blocksize
            need = int(1.5 * self.cfg.audio.samplerate)
            rms, peak = 0.0, 0.0
            crest = float("inf")
            for attempt in range(max(1, attempts)):
                if attempt:
                    time.sleep(2.0)                      # 等 on-demand 麥克風接上
                for _ in range(3):                       # 丟掉開檔暫態
                    stream.read(bs)
                chunks: List[np.ndarray] = []
                got = 0
                while got < need:
                    data, _ = stream.read(bs)
                    arr = np.asarray(data, dtype=np.float64)[:, 0]
                    chunks.append(arr)
                    got += arr.size
                d = np.concatenate(chunks) if chunks else np.zeros(0)
                if d.size:
                    rms = float(np.sqrt(np.mean(np.square(d))))
                    peak = float(np.max(np.abs(d)))
                    crest = peak / rms if rms > 1e-12 else float("inf")
                    if crest >= 2.0 and peak >= 0.003:
                        break
                log.info(
                    "麥克風健檢第 %d 次仍無訊號（crest=%.2f peak=%.5f）→ 2 秒後重試",
                    attempt + 1, crest, peak,
                )
            log.info(
                "麥克風健檢（同一條串流）：rms=%.5f peak=%.5f crest=%.2f",
                rms, peak, crest,
            )
            if crest < 2.0 or peak < 0.003:
                log.warning(
                    "麥克風疑似死訊號（crest=%.2f peak=%.5f）→ 之後拍手不會有反應，"
                    "問題在擷取路徑/驅動，不是拍手門檻。",
                    crest, peak,
                )
            if crest >= 2.0 and peak >= 0.01:
                # 健康檢查通過還不算數：重啟音效堆疊後 mic 常「假活」十幾秒就又凍結，
                # 若一通過就把計數歸零，退避永遠長不起來（2026-09-25 實測：每 2 分鐘
                # 重啟一次、整天 36 次，把使用者的音訊一直打斷）。要連續健康
                # recover_reset_healthy_sec 秒才視為真的恢復。
                now = time.monotonic()
                need = float(
                    getattr(self.cfg.audio, "recover_reset_healthy_sec", 180.0) or 180.0
                )
                if not self._healthy_since:
                    self._healthy_since = now
                elif now - self._healthy_since >= need:
                    self._recover_count = 0
            else:
                self._healthy_since = 0.0
            self._write_status(rms, peak, crest, frozen=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("麥克風健檢失敗：%s", exc)

    def _write_status(self, rms: float, peak: float, crest: float, *, frozen: bool) -> None:
        """寫狀態檔給外部 watchdog 讀（不讓 watchdog 自己去開麥克風）。"""
        path = os.path.expanduser(self.cfg.audio.status_file)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "ts": time.time(),
                        "iso": datetime.now().isoformat(timespec="seconds"),
                        "device": str(self.cfg.audio.device),
                        "rms": rms,
                        "peak": peak,
                        "crest": crest,
                        "frozen": frozen,
                        "ok": (not frozen) and crest >= 2.0 and peak >= 0.01,
                    },
                    fh,
                    ensure_ascii=False,
                )
        except OSError as exc:
            log.debug("狀態檔寫入失敗：%s", exc)

    def _recover_audio(self) -> None:
        """凍結時重啟音訊堆疊。有冷卻，避免把使用者的音訊一直打斷。"""
        base = float(getattr(self.cfg.audio, "recover_cooldown_sec", 0.0) or 0.0)
        if base <= 0:
            return
        # 連續失敗時冷卻倍增（上限 30 分鐘）：mic 真的壞掉時，不要每兩分鐘就打斷
        # 一次使用者的音訊。健康檢查一通過就把計數歸零。
        wait = min(base * (2 ** min(self._recover_count, 5)), 1800.0)
        since = time.monotonic() - self._last_recover
        if self._last_recover and since < wait:
            log.info("距上次音訊恢復 %.0fs（本次冷卻 %.0fs），先只重開串流。", since, wait)
            return
        argv = list(getattr(self.cfg.audio, "recover_command", []) or [])
        if not argv:
            return
        self._last_recover = time.monotonic()
        self._recover_count += 1
        log.warning("執行音訊恢復：%s", " ".join(argv))
        self._write_status(0.0, 0.0, 0.0, frozen=True)
        try:
            subprocess.run(
                argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=60
            )
        except Exception as exc:  # noqa: BLE001
            log.error("音訊恢復失敗：%s", exc)
            return
        time.sleep(float(getattr(self.cfg.audio, "recover_wait_sec", 5.0) or 0.0))
        # 重啟 PipeWire 後同一行程的 PortAudio 會卡住，一定要砍掉重練
        audio.reset_portaudio()
        log.info("音訊恢復完成，重新開啟串流。")

    def _cooldown(self) -> None:
        """誤觸後的冷卻，避免短時間內反覆觸發 STT（每次都要載一次模型）。

        可被 stop 訊號打斷。
        """
        seconds = float(getattr(self.cfg.wake, "cooldown_sec", 0.0) or 0.0)
        if seconds <= 0:
            return
        import time

        deadline = time.monotonic() + seconds
        while not self._stop and time.monotonic() < deadline:
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # R2 + R3 + R4：錄需求 → 轉錄 → 回述確認
    # ------------------------------------------------------------------
    def _cue_start(self, attempt: int = 0) -> None:
        """「開始聽」提示。

        `tts.prompt_mode == "chime"`（使用者 2026-09-25 指定的預設）：
        只播一次提示音，完全不講話——講話的提示音太慢也太吵。
        音檔用 `cue="start"` 那一組（開頭那顆，通常較高）。
        `"voice"`：舊行為（可選 beep + TTS 引導語，重問時換一句）。
        """
        if self.cfg.tts.prompt_mode == "chime":
            log.info("提示音：開始（cue=start）")
            tts.play_chime(self.cfg, logger=log, cue="start")
            return
        if self.cfg.tts.beep_enabled:
            tts.play_beep(self.cfg, logger=log)
        if attempt:
            lines = self.cfg.tts.retry_prompts or [self.cfg.tts.ok_prompt]
            tts.speak(lines[(attempt - 1) % len(lines)], self.cfg, logger=log)
        else:
            tts.speak(self.cfg.tts.ok_prompt, self.cfg, logger=log)

    def _cue_end(self) -> None:
        """「聽完了」提示：chime 模式下再播一次提示音（`cue="end"`，通常較低）。

        有聽到需求、或使用者根本沒講話（前置靜音逾時）都算一輪結束，
        所以兩種情況都會響——使用者要的是「開始一聲、結束一聲」的節奏，
        而且兩聲要**明顯不同**才不會搞錯現在是哪個階段。
        """
        if self.cfg.tts.prompt_mode == "chime":
            log.info("提示音：結束（cue=end）")
            tts.play_chime(self.cfg, logger=log, cue="end")

    def prompt_and_capture(self, stream, attempt: int = 0) -> Optional[str]:
        """播提示音（chime＝咚咚／voice＝TTS）→ 錄需求 → 播結束音 → STT。

        attempt=0 講第一次的引導語；attempt>0 走「重問」那一組，
        每次換一句（不要像機器人一樣重播同一句）。
        回傳轉錄字串或 None。
        """
        self._cue_start(attempt)
        samples = self._record_utterance(stream)
        self._cue_end()
        if samples is None or samples.size == 0:
            log.info("沒有錄到語音（前置靜音逾時）→ 第 %d 次嘗試", attempt + 1)
            return None
        dur = samples.size / float(self.cfg.audio.samplerate)
        log.info("需求錄音：%.2fs（第 %d 次嘗試）→ 送 STT", dur, attempt + 1)
        try:
            transcript = stt.transcribe_samples(samples, self.cfg, logger=log)
        except stt.SttError as exc:
            log.error("需求 STT 失敗：%s", exc)
            return None
        text = transcript.strip()
        if not text:
            log.warning("需求 STT 回空字串（錄到 %.2fs）→ 當作沒聽到", dur)
            return None
        # STT 腳本會把「疑似幻覺」的結果加上 ⚠️ 前綴；這種文字**絕不能派工**，
        # 否則會送出一張內容是「謝謝觀看,下次見。」之類的假任務
        # （2026-09-25 實際發生：假任務卡跑到 #人工智障）。
        if text.startswith("⚠️") or "疑似辨識幻覺" in text:
            log.warning("STT 標記為疑似幻覺 → 不派工（%r）", text[:60])
            return None
        log.info("需求轉錄：%r", text)
        return text

    def _is_retry_only(self, transcript: str) -> bool:
        """整句就只是否定／要求重說嗎？

        刻意要求「短」：disagree_words 裡有長度 1 的「不」，
        若只看有沒有命中，「我覺得這不行」也會被當成否定而白重錄一次。
        """
        norm = normalize(transcript)
        if not norm or len(norm) > 5:
            return False
        return (
            classify_confirmation(norm, [], self.cfg.confirm.disagree_words)
            == "disagree"
        )

    def _looks_like_own_prompt(self, transcript: str) -> bool:
        """轉錄內容是不是我們自己的提示語（回音殘留）？

        縱使有 settle/drain，偶爾仍可能錄到自己的尾音；一旦把自己的台詞當成
        需求派出去就是一個假任務（2026-09-25 實際發生：需求原文變成
        「這次說大聲一點。」「我在聽。NO NO NO」）。
        """
        norm = normalize(transcript)
        if len(norm) < 3:
            return False
        tts_cfg = self.cfg.tts
        own = [tts_cfg.ok_prompt, tts_cfg.unclear_prompt, tts_cfg.give_up_prompt,
               tts_cfg.dispatched_prompt, *tts_cfg.retry_prompts]
        return any(norm == normalize(p) or norm in normalize(p) for p in own if p)

    def record_and_confirm(self, stream) -> Optional[str]:
        """錄需求 → STT → 複述（告知用，不等回覆）→ 交付。

        2026-09-25 改版：拿掉「必須說『對』才算數」的確認輪。
        舊版固定錄 3 秒等使用者說「對」，但「對」只有約 0.3 秒，VAD 幾乎把
        它全砍掉 → Whisper 回吐幻覺（實測連續得到「然後想說,造孽啊。」
        「作為成功我肯定會破防。」）→ 使用者卡在鬼打牆、一次互動要一分鐘。

        改成 fail-open：只有整句就是否定詞（不／錯／重來／再說）才重錄，
        其他一律直接派工；複述只是讓使用者知道聽到什麼，不等待回覆。
        """
        retries = 0
        while retries <= self.cfg.confirm.max_retries and not self._stop:
            transcript = self.prompt_and_capture(stream, attempt=retries)
            if not transcript:
                # 沒收到錄音（沒講話／VAD 逾時）：使用者要「就翻過來咚懂、然後結束」，
                # 不要在那裡反覆重問（重問會多出一堆提示音）。預設不重試。
                if not self.cfg.confirm.retry_on_no_speech:
                    log.info("沒收到錄音 → 結束本輪（不重試；結束音已響過）")
                    break
                retries += 1
                continue
            if self._looks_like_own_prompt(transcript):
                log.warning("轉錄像我們自己的提示語（%r）→ 疑似回音，當作沒聽到", transcript)
                retries += 1
                continue
            if self._is_retry_only(transcript):
                log.info("需求只有否定／重來詞（%r）→ 重錄。", transcript)
                retries += 1
                continue
            if self.cfg.tts.prompt_mode != "chime" and self.cfg.tts.confirm_template:
                tts.speak(
                    self.cfg.tts.confirm_template.format(transcript=transcript),
                    self.cfg, logger=log,
                )
            return transcript
        # chime 模式下「結束咚」已經在 prompt_and_capture 響過，這裡不再多出聲
        if self.cfg.tts.prompt_mode != "chime":
            tts.speak(self.cfg.tts.give_up_prompt, self.cfg, logger=log)
        log.info("超過重試上限，放棄本輪。")
        return None

    # ------------------------------------------------------------------
    # R5 + R6：轉發 Discord + 派工
    # ------------------------------------------------------------------
    def _add_thread_member(self, client, thread_id: str) -> None:
        """把使用者加入討論串（2026-09-26 使用者：「自動把我加到串裏面」）。

        語音派工的討論串是 bot 開的，使用者預設不在成員名單裡、收不到通知；
        建立後補一發 PUT thread-members 把他加進去。

        **非致命**：沒設 user_id、或權限不足／42000 之類的失敗，只記 warning，
        不影響派工（討論串本身仍然有效）。
        """
        uid = str(getattr(self.cfg.discord, "user_id", "") or "").strip()
        if not uid:
            return
        try:
            client.add_thread_member(thread_id, uid)
            log.info("已把使用者 %s 加入討論串 %s", uid, thread_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("加入討論串成員失敗（非致命，略過）：%s", exc)

    def _bot_identity(self, client) -> dict:
        """要模仿的發話身分＝Hermes bot 自己（`GET /users/@me`），只抓一次。

        2026-09-26 使用者：「要模仿 bot 在伺服器的外觀」——webhook 訊息顯示的是
        發話者的名稱＋頭像，直接沿用 bot 自己的，看起來就跟 bot 發的一樣。
        抓不到就回空 dict（退回設定裡的 `relay.username`）。
        """
        cached = getattr(self, "_bot_identity_cache", None)
        if cached is not None:
            return cached
        ident: dict = {}
        try:
            me = client.get_self()
            ident = {
                "username": str(me.get("username") or ""),
                "avatar_url": user_avatar_url(me),
            }
            log.info("relay 將模仿 bot 身分：%s", ident.get("username") or "(讀不到名稱)")
        except Exception as exc:  # noqa: BLE001
            log.warning("讀取 bot 身分失敗（%s）→ 改用 relay.username", exc)
        self._bot_identity_cache = ident
        return ident

    def _await_thread(self, client, message_id: str, timeout_sec: float = 8.0) -> str:
        """等 gateway 把那則訊息變成討論串（`force_thread_channels`），回傳 thread id。

        輪詢 `GET /channels/{ch}/messages/{mid}` 的 `thread` 欄位。由訊息長出來的
        討論串 **thread id == message id**，所以一掛上就知道。等不到回 **空字串**
        （呼叫端會當成 relay 失敗 → 回退 spawn）；8 秒是刻意留的餘裕：實測 gateway
        開串只要 0.3 秒，等不到通常就是「gateway 根本沒接手」（例如 @ 閘沒過）。
        """
        deadline = time.time() + max(0.0, timeout_sec)
        channel_id = self.cfg.discord.channel_id
        while time.time() < deadline:
            try:
                m = client.get_message(channel_id, message_id)
                tid = str(((m.get("thread") or {}) or {}).get("id") or "")
                if tid:
                    return tid
            except Exception as exc:  # noqa: BLE001
                log.warning("輪詢討論串失敗（%s）", exc)
            time.sleep(0.5)
        log.warning("等不到 gateway 掛上討論串（訊息 %s）→ 當 relay 失敗，回退 spawn",
                    message_id)
        return ""

    def forward_and_dispatch(self, transcript: str) -> None:
        """把需求轉發到 Discord、開 thread、派工 hermes 並回報。"""
        now = datetime.now()
        thread_name = make_thread_name(
            transcript,
            now.strftime("%m-%d %H:%M"),
            template=self.cfg.discord.thread_name_template,
            short_len=self.cfg.discord.thread_short_len,
            limit=self.cfg.discord.thread_name_limit,
        )
        card = self.cfg.discord.card_template.format(transcript=transcript,
                                                     date=now.strftime("%Y-%m-%d %H:%M:%S"))
        task_card = self.cfg.discord.task_card_template.format(
            transcript=transcript,
            date=now.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if self.dry_run:
            self._print_dry_run(transcript, thread_name, card, task_card)
            return

        client = DiscordClient(self.cfg, logger=log)
        channel_id = self.cfg.discord.channel_id

        relay = getattr(self.cfg, "relay", None)
        use_relay = bool(
            relay and getattr(relay, "enabled", False) and getattr(self.cfg, "relay_url", "")
        )
        thread_id = ""      # 兩條路徑都會設；先給值讓靜態檢查與例外路徑都安全
        message_id = ""

        # 2026-09-26 定案：「能直接用 webhook 發嗎？節省一則訊息，而且要模仿 bot 在
        # 伺服器的外觀」。實測發現 gateway 的 `force_thread_channels` 對 #人工智障 的
        # 訊息**強制開串** → 只要 webhook 在母頻道發一則「<@Hermes> 需求原文」，
        # gateway 就會自己把它變成討論串的開頭並在串內處理（session key = thread id
        # == message id）。所以不必自己發卡、也不必再補一則串內觸發訊息：全場只有
        # 這一則，需求只出現一次、外面看得到。
        if use_relay:
            try:
                ident = self._bot_identity(client)
                sent = dispatch.post_relay(
                    transcript, self.cfg, logger=log, identity=ident
                )
                message_id = str(sent.get("id", ""))
                if not message_id:
                    raise RuntimeError(f"relay 未回傳 message id：{sent}")
                thread_id = str(((sent.get("thread") or {}) or {}).get("id") or "")
                if not thread_id:      # gateway 開串要一點時間 → 輪詢
                    thread_id = self._await_thread(client, message_id)
                if not thread_id:
                    raise RuntimeError(
                        f"gateway 沒把訊息 {message_id} 開成討論串"
                        "（relay.mention_id 留空但 .env 的 DISCORD_ALLOW_BOTS 還是 mentions？）")
                log.info(
                    "需求已由 webhook 發在母頻道（msg=%s）；gateway 討論串 %s",
                    message_id, thread_id,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("relay 路徑失敗：%s → 回退 spawn hermes（自行開串）", exc)
                # 把已經發出去、但沒人接手的訊息收回來，免得母頻道留一則孤兒。
                if message_id:
                    try:
                        client.delete_message(channel_id, message_id)
                        log.info("已刪掉沒人接手的 relay 訊息 %s", message_id)
                    except Exception as inner:  # noqa: BLE001
                        log.warning("刪除 relay 訊息失敗（非致命）：%s", inner)
                    message_id = ""
                use_relay = False

        if not use_relay:
            # 回退路徑：relay 停用或失敗才走這裡——bot 發卡 → 由該訊息長出討論串。
            log.info("發送需求到主頻道 %s（回退路徑：自行開串）", channel_id)
            msg = client.post_message(channel_id, card)
            message_id = str(msg.get("id", ""))
            if not message_id:
                raise RuntimeError(f"Discord 未回傳 message id：{msg}")
            log.info("建立討論串：%s", thread_name)
            thread = client.create_thread_from_message(
                channel_id, message_id, thread_name
            )
            thread_id = str(thread.get("id", ""))
            if not thread_id:
                raise RuntimeError(f"Discord 未回傳 thread id：{thread}")

        # 把使用者加進討論串（讓他收到通知、可以直接在串內回話）。
        # relay 路徑的串是 **gateway 開的**，他一定不在成員名單裡；回退路徑同理。
        self._add_thread_member(client, thread_id)

        # 串內任務卡：**留空＝不發**（2026-09-26 使用者：「幾則就好」）。
        if task_card.strip():
            client.post_thread_message(thread_id, task_card)

        prompt = dispatch.build_dispatch_prompt(
            transcript, channel_id, thread_id, self.cfg
        )

        # 保底機制：派工程序結束時，若 agent **完全沒貼任何訊息到串內**，就把它的
        # 最終 stdout 貼上去。（2026-09-26 使用者：「他有做事，但都沒輸出到 dc，
        # 看起來就沒有」——不能只依賴 agent 記得自己跑 `hermes send`。）
        try:
            _before = {str(m.get("id")) for m in _as_list(client.get_messages(thread_id, 20))}
        except Exception as exc:  # noqa: BLE001
            log.warning("讀取串內既有訊息失敗（保底去重會失效）：%s", exc)
            _before = set()

        def _on_exit(_proc, log_file):
            text = dispatch.extract_stdout(log_file)
            if not text:
                return
            # 1) 語音回報：做好後用「講的」把結果告訴使用者（2026-09-26 要求）
            if getattr(self.cfg.tts, "speak_result", True):
                try:
                    summary = spoken_summary(
                        text,
                        int(getattr(self.cfg.tts, "speak_result_max_chars", 160) or 160),
                    )
                    if summary:
                        log.info("語音回報結果：%s", summary[:80])
                        tts.speak(summary, self.cfg, logger=log)
                except Exception as exc:  # noqa: BLE001
                    log.warning("語音回報失敗：%s", exc)
            # 2) 保底把最終輸出貼回串（若 agent 沒自己貼過任何訊息）
            try:
                c2 = DiscordClient(self.cfg, logger=log)
                after = {str(m.get("id")) for m in _as_list(c2.get_messages(thread_id, 20))}
                if after - _before:
                    log.info("派工已自行回報（%d 則）→ 不補貼 stdout", len(after - _before))
                    return
                body = text if len(text) <= 1900 else text[:1900] + "\n…（截斷）"
                c2.post_thread_message(thread_id, f"📄 **最終輸出**（agent 未自行回報）：\n{body}")
                log.info("agent 未回報 → 已把最終輸出補貼到串 %s", thread_id)
            except Exception as exc:  # noqa: BLE001
                log.error("補貼最終輸出失敗：%s", exc)

        def _heartbeat(secs):
            """派工還在跑 → 在串內發一則心跳，讓使用者知道有在動。"""
            try:
                m, s = divmod(int(secs), 60)
                human = f"{m} 分 {s} 秒" if m else f"{s} 秒"
                DiscordClient(self.cfg, logger=log).post_thread_message(
                    thread_id, f"⏳ 仍在處理中…（已 {human}）"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("心跳發送失敗：%s", exc)

        def _watch_thread_for_result():
            """relay 模式專用：agent 交給 gateway 跑，我們讀不到它的 stdout。

            2026-09-26 使用者定案：
              「TTS 合成加速，現在跑完還要等一下才能聽到」
              「只有回報要說，中間思考、做事不要說話」
              「為什麼有些任務明明還沒跑完卻直接回覆我，而且是進度不是結果」→ 選 (c)
            → 讀 Hermes 的 session DB（~/.hermes/state.db）。一個**真正結束**的回合實測長這樣：

                  role=assistant, finish_reason='stop'   ← 最終回覆
                  role=session_meta                      ← 回合收尾列

            (c) ＝**安靜緩衝 ＋ 重新武裝**：
              - 安靜緩衝（`speak_result_settle_sec`，預設 20s）：看到 session_meta 後，還要
                「這 N 秒內 DB 都沒有新訊息」才唸。避免 agent 只是換口氣、馬上又繼續跑，
                就把中間那則當結果唸掉。
              - 重新武裝：唸完**不結束**，繼續盯；之後又有新回合結束就再唸一次 →
                最終結果一定聽得到（實測踩到的情境：22:05 結束第一回合 → 使用者 22:09
                才追加 → 22:18 才有最終結果，舊版唸完第一則就收工了）。
                上限：同一個串最多唸 `speak_result_max_speaks`（3）次、
                總共只盯 `speak_result_watch_sec`（1800s＝30 分），免得在串內閒聊被唸。
            讀不到 DB／20 秒內找不到對應 session 時，退回舊的「等安靜 quiet 秒讀串」（單次）。
            """
            import threading as _th

            quiet_needed = float(getattr(self.cfg.tts, "speak_result_quiet_sec", 8.0) or 8.0)
            poll_interval = float(getattr(self.cfg.tts, "speak_result_poll_sec", 2.0) or 2.0)
            max_chars = int(getattr(self.cfg.tts, "speak_result_max_chars", 160) or 160)
            settle_sec = float(getattr(self.cfg.tts, "speak_result_settle_sec", 20.0) or 20.0)
            max_speaks = int(getattr(self.cfg.tts, "speak_result_max_speaks", 3) or 3)
            watch_sec = float(
                getattr(self.cfg.tts, "speak_result_watch_sec", 1800.0) or 1800.0)
            db_path = os.path.expanduser(
                getattr(self.cfg.tts, "speak_result_session_db", "~/.hermes/state.db"))
            deadline = time.time() + watch_sec

            def _speak(text: str) -> bool:
                try:
                    summary = spoken_summary(text, max_chars)
                    if not summary:
                        return False
                    log.info("語音回報結果：%s", summary[:80])
                    tts.speak(summary, self.cfg, logger=log)
                    return True
                except Exception as exc:  # noqa: BLE001
                    log.warning("語音回報失敗：%s", exc)
                    return False

            def _is_notice(text: str) -> bool:
                """串內「非 agent 最終回覆」的訊息（公告／心跳／relay 原文）——不該被唸。"""
                t = text.strip()
                return (
                    not t
                    or "<@" in t
                    or t.startswith("🎙️")
                    or t.startswith("✅ 收到")
                    or t.startswith("⏳")
                    or t.startswith("⚠️")
                    or "仍在處理中" in t
                )

            def _db_watch() -> Optional[bool]:
                """盯 session DB：回合結束＋安靜 settle 秒 → 唸最終回覆；唸完**繼續盯**。

                (c) 的行為：命中一次之後不結束，重新武裝等下一個回合（上限 max_speaks 次、
                最多盯 deadline）。回傳 True＝至少唸過一次（成功）；
                False＝此路不通（沒 session／讀不到），改走讀串備援。
                """
                import sqlite3
                try:
                    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
                    con.execute("pragma query_only=1")
                except Exception as exc:  # noqa: BLE001
                    log.warning("開不了 session DB（%s）→ 退回讀串判定", exc)
                    return False
                sid, since = "", 0
                spoken = 0              # 已唸次數（上限 max_speaks）
                last_spoken_id = 0      # 已唸到哪一則（同一則不重複唸）
                last_activity = time.time()
                find_until = time.time() + 20.0
                try:
                    while time.time() < deadline and not self._stop:
                        if not sid:
                            row = con.execute(
                                "select id from sessions where (thread_id=? or chat_id=?) "
                                "order by started_at desc limit 1",
                                (thread_id, thread_id)).fetchone()
                            if row:
                                sid = row[0]
                                since = int(con.execute(
                                    "select coalesce(max(id),0) from messages where session_id=?",
                                    (sid,)).fetchone()[0])
                                log.info(
                                    "語音回報：鎖定 session %s（等回合結束＋安靜 %.0f 秒，"
                                    "最多唸 %d 次）", sid, settle_sec, max_speaks)
                            elif time.time() > find_until:
                                log.info("語音回報：找不到對應 session → 退回讀串判定")
                                return False
                            time.sleep(poll_interval)
                            continue

                        rows = list(con.execute(
                            "select id, role, coalesce(content,''), coalesce(finish_reason,'') "
                            "from messages where session_id=? and id>? order by id",
                            (sid, since)))
                        if rows:
                            since = max(since, rows[-1][0])
                            last_activity = time.time()

                        # 條件：至少有一個回合收尾列（session_meta）＋ 這 settle 秒內 DB 沒動靜
                        finished = bool(con.execute(
                            "select 1 from messages where session_id=? and role='session_meta' "
                            "limit 1", (sid,)).fetchone())
                        if finished and (time.time() - last_activity) >= settle_sec:
                            pick = pick_report(
                                list(con.execute(
                                    "select id, role, coalesce(content,''), "
                                    "coalesce(finish_reason,'') from messages "
                                    "where session_id=? and id>? and role='assistant' "
                                    "and coalesce(finish_reason,'')='stop' order by id",
                                    (sid, last_spoken_id))),
                                last_spoken_id,
                            )
                            if pick:
                                mid, content = pick
                                if _speak(content):
                                    spoken += 1
                                    last_spoken_id = mid
                                    last_activity = time.time()
                                    log.info("語音回報：第 %d/%d 次唸完（msg=%s）→ 繼續盯",
                                             spoken, max_speaks, mid)
                                    if spoken >= max_speaks:
                                        return True
                                else:
                                    return False
                        time.sleep(poll_interval)
                except Exception as exc:  # noqa: BLE001
                    log.warning("讀 session DB 失敗（%s）→ 退回讀串判定", exc)
                    return bool(spoken)
                finally:
                    try:
                        con.close()
                    except Exception:  # noqa: BLE001
                        pass
                return bool(spoken)

            def _thread_watch():
                """備援：舊的「等安靜 quiet 秒 → 唸串內最後一則 agent 訊息」。"""
                # 建構就可能失敗（沒 token／設定缺）→ 不能讓它把背景執行緒炸掉
                # （實測：這會變成未捕捉的背景例外，只在 log 裡留一串 traceback）。
                try:
                    c = DiscordClient(self.cfg, logger=log)
                except Exception as exc:  # noqa: BLE001
                    log.warning("讀串備援路徑無法連 Discord（%s）→ 放棄語音回報", exc)
                    return
                if self._stop:
                    return
                last_seen = ""
                last_change = time.time()
                bot_id = str(getattr(self.cfg.relay, "mention_id", ""))
                while time.time() < deadline and not self._stop:
                    try:
                        msgs = _as_list(c.get_messages(thread_id, 20))
                    except Exception:  # noqa: BLE001
                        time.sleep(10)
                        continue
                    newest = max((str(m.get("id", "")) for m in msgs), default="")
                    if newest and newest != last_seen:
                        last_seen = newest
                        last_change = time.time()
                    elif last_seen and (time.time() - last_change) >= quiet_needed:
                        for m in msgs:
                            author_id = str((m.get("author") or {}).get("id", ""))
                            content = str(m.get("content", ""))
                            if bot_id and author_id != bot_id:
                                continue
                            if _is_notice(content):
                                continue
                            _speak(content.strip())
                            return
                        return
                    time.sleep(poll_interval)

            def _loop():
                if _db_watch() is True:
                    return
                _thread_watch()

            _th.Thread(target=_loop, daemon=True).start()

        # ── 派工方式 ──────────────────────────────────────────────────
        #   relay（預設）：需求那一則**已經**在上方由 webhook 發到母頻道，gateway 自己
        #     開串接手 → 這裡只要開始盯結果（讀 session DB → 念最終回覆）。
        #   spawn（回退）：relay 停用或失敗時，才由我們自己 spawn `hermes -z`。
        if use_relay:
            # 派工通知：留空＝不發（gateway 接手後自己會回一則，這則是多餘的）。
            if self.cfg.discord.dispatched_notice.strip():
                client.post_thread_message(thread_id, self.cfg.discord.dispatched_notice)
            if self.cfg.tts.prompt_mode != "chime":
                tts.speak(self.cfg.tts.dispatched_prompt, self.cfg, logger=log)
            _watch_thread_for_result()
            log.info("relay 派工完成（gateway 接手），thread=%s", thread_id)
            return

        try:
            dispatch.spawn_hermes(
                prompt, self.cfg, thread_id, logger=log, on_exit=_on_exit,
                heartbeat=_heartbeat,
                heartbeat_sec=float(getattr(self.cfg.dispatch, "heartbeat_sec", 0.0) or 0.0),
            )
        except Exception as exc:  # noqa: BLE001 - 派工失敗要讓串內看得見
            log.error("派工失敗：%s", exc)
            try:
                client.post_thread_message(
                    thread_id, f"⚠️ 派工失敗，請看 daemon log：{exc}"
                )
            except Exception as inner:  # noqa: BLE001
                log.error("連失敗通知都發不出去：%s", inner)
            return
        if self.cfg.discord.dispatched_notice.strip():
            client.post_thread_message(thread_id, self.cfg.discord.dispatched_notice)
        # chime 模式：整場只有「開始咚／結束咚」兩聲，派工後不再出聲
        if self.cfg.tts.prompt_mode != "chime":
            tts.speak(self.cfg.tts.dispatched_prompt, self.cfg, logger=log)
        log.info("派工完成，thread=%s", thread_id)

    def _print_dry_run(self, transcript, thread_name, card, task_card) -> None:
        prompt = dispatch.build_dispatch_prompt(
            transcript, self.cfg.discord.channel_id, "<thread_id>", self.cfg
        )
        argv = dispatch.build_hermes_command(prompt, self.cfg)
        # dry-run 不載入 .env，所以拿不到 relay_url；這裡跟下方「派工方式」一致，
        # 只看 relay.enabled。
        relay_on = bool(getattr(self.cfg.relay, "enabled", False))
        lines = [
            "",
            "===== DRY RUN（不會真的發送 Discord 或 spawn hermes）=====",
            f"[目標母頻道] {self.cfg.discord.channel_id}",
            "[派工方式]",
            (f"  relay（全場 1 則）：webhook 直接在母頻道發\n"
             f"    「{dispatch.build_relay_content(transcript, self.cfg)}」\n"
             "    （模仿 bot 的名字＋頭像。gateway 的 force_thread_channels 會自己替它"
             "開討論串\n     並在串內處理 → 需求只出現一次、外面看得到）\n"
             "    webhook URL 讀 .env，dry-run 不載入 secrets"
             if relay_on
             else "  spawn（relay 停用／失敗的回退路徑）：\n"
                  f"    母頻道卡「{card}」→ 自行開討論串「{thread_name}」→\n"
                  f"    {argv[0]} {argv[1]} <dispatch_prompt>"),
            "[加入討論串成員] user_id="
            f"{getattr(self.cfg.discord, 'user_id', '') or '（未設定，不加入）'}",
            "[串內任務卡／派工完成通知] "
            f"{task_card.strip() or self.cfg.discord.dispatched_notice.strip() or '（留空＝不發）'}",
            "[dispatch_prompt 內容（只有回退 spawn 路徑會用到）]",
            prompt,
            "==========================================================",
            "",
        ]
        print("\n".join(lines))

    # ------------------------------------------------------------------
    # 對外進入點
    # ------------------------------------------------------------------
    def simulate(self, text: str) -> int:
        """跳過麥克風，直接以 text 走 R5/R6（供測試 / 無麥克風）。"""
        text = (text or "").strip()
        if not text:
            log.error("--simulate 需要非空文字")
            return 2
        log.info("SIMULATE 模式，需求文字：%r（略過語音確認）", text)
        try:
            self.forward_and_dispatch(text)
        except Exception as exc:  # noqa: BLE001
            log.error("轉發/派工失敗：%s", exc)
            return 1
        return 0

    def run_once(self) -> int:
        """完整跑一輪 R1–R6（需要麥克風）。"""
        try:
            if not self.wait_for_wake():
                return 0
            with audio.input_stream(self.cfg.audio) as stream:
                transcript = self.record_and_confirm(stream)
            if not transcript:
                return 0
            self.forward_and_dispatch(transcript)
        except audio.AudioUnavailable as exc:
            log.error("音訊裝置不可用：%s", exc)
            return 3
        except Exception as exc:  # noqa: BLE001
            log.error("本輪執行失敗：%s", exc)
            return 1
        return 0

    def run(self) -> int:
        """常駐迴圈，直到收到停止訊號。"""
        self.install_signal_handlers()
        log.info("hermes-voice-dispatch 啟動，開始監聽…")
        # 把實際解析到的輸入裝置寫進 log：裝置指錯（例如指到收不到聲音的
        # 節點）時，症狀會是「在跑但永遠沒反應」，有這行才好查。
        log.info("輸入裝置解析結果：%s", audio.describe_device(self.cfg.audio.device))
        # 注意：健檢「不可以另外開一條 stream」。本機（acer-ubuntu）的 DMIC 多開／
        # 重開常常只拿到凍結的直流，而正常的開檔只有一次；另外開一條去健檢會把
        # 那次搶走，害真正在監聽的那條變成死的（2026-09-25 實際踩到）。
        # 所以健檢改成在 wait_for_wake() 的同一條串流上做。
        try:
            failures = 0
            while not self._stop:
                rc = self.run_once()
                if rc == 3:  # 音訊裝置不可用，沒必要空轉
                    return rc
                if rc != 0:
                    # 一定要退避！否則音訊掛掉時這裡會變成忙迴圈，把 journal 灌爆
                    failures += 1
                    delay = min(60.0, 2.0 ** min(failures, 5))
                    log.warning("本輪失敗（rc=%s），%.0fs 後重試。", rc, delay)
                    audio.reset_portaudio()
                    self._sleep_interruptible(delay)
                else:
                    failures = 0
        except audio.AudioUnavailable as exc:
            log.error("音訊裝置不可用：%s", exc)
            return 3
        log.info("已停止。")
        return 0

    def _sleep_interruptible(self, seconds: float) -> None:
        """睡一段時間，可被 stop 訊號打斷。"""
        deadline = time.monotonic() + max(0.0, seconds)
        while not self._stop and time.monotonic() < deadline:
            time.sleep(0.2)
