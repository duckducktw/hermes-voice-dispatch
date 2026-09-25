"""主狀態機：idle → wake → record → confirm → forward → dispatch。

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
from datetime import datetime
from typing import List, Optional

import numpy as np

from . import audio, dispatch, stt, tts, wake
from .config import Config
from .discord_api import DiscordClient
from .text import classify_confirmation, make_thread_name
from .vad import VadSegmenter, VadState

log = logging.getLogger("voice_dispatch")


class VoiceDispatcher:
    def __init__(self, cfg: Config, *, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self._stop = False
        self._last_recover = 0.0
        self._recover_count = 0
        self._healthy_since = 0.0   # 連續健康起算點（見 _log_stream_health）

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

    def _record_utterance(self, stream) -> Optional[np.ndarray]:
        """能量 VAD 錄一段話。回傳樣本；前置靜音逾時回傳 None。"""
        seg = VadSegmenter(self.cfg.vad)
        blocksize = self.cfg.audio.blocksize
        samplerate = self.cfg.audio.samplerate
        block_dur = blocksize / float(samplerate)
        chunks: List[np.ndarray] = []
        idx = 0
        for block in audio.read_blocks(stream, blocksize):
            if self._stop:
                break
            t = idx * block_dur
            idx += 1
            rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
            state = seg.feed(rms, t)
            if seg.started:
                chunks.append(block)
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
    def wait_for_wake(self) -> bool:
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
                    blocks, self.cfg, should_stop=lambda: self._stop
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
                        transcript = stt.transcribe_samples(
                            samples, self.cfg, logger=log
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

    def _log_stream_health(self, stream) -> None:
        """在「同一條」串流上量 crest，把「死訊號」講清楚。

        不要為了健檢另外開 stream：本機多開會拿到凍結訊號，反而害到監聽中的
        那條（2026-09-25 實際踩到的坑）。
        """
        try:
            bs = self.cfg.audio.blocksize
            for _ in range(3):  # 丟掉開檔暫態
                stream.read(bs)
            need = int(1.5 * self.cfg.audio.samplerate)
            chunks: List[np.ndarray] = []
            got = 0
            while got < need:
                data, _ = stream.read(bs)
                arr = np.asarray(data, dtype=np.float64)[:, 0]
                chunks.append(arr)
                got += arr.size
            d = np.concatenate(chunks) if chunks else np.zeros(0)
            if d.size == 0:
                return
            rms = float(np.sqrt(np.mean(np.square(d))))
            peak = float(np.max(np.abs(d)))
            crest = peak / rms if rms > 1e-12 else float("inf")
            log.info(
                "麥克風健檢（同一條串流）：rms=%.5f peak=%.5f crest=%.2f",
                rms, peak, crest,
            )
            if crest < 2.0 or peak < 0.01:
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
    def prompt_and_capture(self, stream) -> Optional[str]:
        """播提示音 + TTS 引導 → 錄需求 → STT。回傳轉錄字串或 None。"""
        tts.play_beep(self.cfg, logger=log)
        tts.speak(self.cfg.tts.ok_prompt, self.cfg, logger=log)
        samples = self._record_utterance(stream)
        if samples is None or samples.size == 0:
            log.info("沒有錄到語音（前置靜音逾時）。")
            return None
        try:
            transcript = stt.transcribe_samples(samples, self.cfg, logger=log)
        except stt.SttError as exc:
            log.error("需求 STT 失敗：%s", exc)
            return None
        return transcript.strip() or None

    def confirm(self, stream, transcript: str) -> str:
        """回述確認一次。回傳 "agree" / "disagree" / "unknown"。"""
        prompt = self.cfg.tts.confirm_template.format(transcript=transcript)
        tts.speak(prompt, self.cfg, logger=log)
        samples = self._collect_seconds(stream, self.cfg.confirm.record_sec)
        try:
            answer = stt.transcribe_samples(samples, self.cfg, logger=log)
        except stt.SttError as exc:
            log.warning("確認 STT 失敗：%s", exc)
            return "unknown"
        log.info("確認回覆轉錄：%r", answer)
        return classify_confirmation(
            answer, self.cfg.confirm.agree_words, self.cfg.confirm.disagree_words
        )

    def record_and_confirm(self, stream) -> Optional[str]:
        """R2–R4 完整流程；回傳已確認的需求文字或 None（放棄）。"""
        retries = 0
        while retries <= self.cfg.confirm.max_retries and not self._stop:
            transcript = self.prompt_and_capture(stream)
            if not transcript:
                retries += 1
                continue

            unclear = 0
            while not self._stop:
                verdict = self.confirm(stream, transcript)
                if verdict == "agree":
                    return transcript
                if verdict == "disagree":
                    log.info("使用者不同意，重新錄需求。")
                    break  # 回到外層重錄
                # unknown
                unclear += 1
                if unclear >= self.cfg.confirm.max_unclear:
                    tts.speak(self.cfg.tts.give_up_prompt, self.cfg, logger=log)
                    return None
                tts.speak(self.cfg.tts.unclear_prompt, self.cfg, logger=log)
            retries += 1
        log.info("超過重試上限，放棄本輪。")
        return None

    # ------------------------------------------------------------------
    # R5 + R6：轉發 Discord + 派工
    # ------------------------------------------------------------------
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
        card = self.cfg.discord.card_template.format(transcript=transcript)
        task_card = self.cfg.discord.task_card_template.format(
            transcript=transcript,
            date=now.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if self.dry_run:
            self._print_dry_run(transcript, thread_name, card, task_card)
            return

        client = DiscordClient(self.cfg, logger=log)
        channel_id = self.cfg.discord.channel_id

        log.info("發送語音派工卡到主頻道 %s", channel_id)
        msg = client.post_message(channel_id, card)
        message_id = str(msg.get("id", ""))
        if not message_id:
            raise RuntimeError(f"Discord 未回傳 message id：{msg}")

        log.info("建立討論串：%s", thread_name)
        thread = client.create_thread_from_message(channel_id, message_id, thread_name)
        thread_id = str(thread.get("id", ""))
        if not thread_id:
            raise RuntimeError(f"Discord 未回傳 thread id：{thread}")

        client.post_thread_message(thread_id, task_card)

        prompt = dispatch.build_dispatch_prompt(
            transcript, channel_id, thread_id, self.cfg
        )
        try:
            dispatch.spawn_hermes(prompt, self.cfg, thread_id, logger=log)
        except Exception as exc:  # noqa: BLE001 - 派工失敗要讓串內看得見
            log.error("派工失敗：%s", exc)
            try:
                client.post_thread_message(
                    thread_id, f"⚠️ 派工失敗，請看 daemon log：{exc}"
                )
            except Exception as inner:  # noqa: BLE001
                log.error("連失敗通知都發不出去：%s", inner)
            return
        client.post_thread_message(thread_id, self.cfg.discord.dispatched_notice)
        tts.speak(self.cfg.tts.dispatched_prompt, self.cfg, logger=log)
        log.info("派工完成，thread=%s", thread_id)

    def _print_dry_run(self, transcript, thread_name, card, task_card) -> None:
        prompt = dispatch.build_dispatch_prompt(
            transcript, self.cfg.discord.channel_id, "<thread_id>", self.cfg
        )
        argv = dispatch.build_hermes_command(prompt, self.cfg)
        lines = [
            "",
            "===== DRY RUN（不會真的發送 Discord 或 spawn hermes）=====",
            f"[目標主頻道] {self.cfg.discord.channel_id}",
            f"[語音派工卡] {card}",
            f"[討論串名稱] {thread_name}（長度 {len(thread_name)}）",
            "[討論串任務卡]",
            task_card,
            "[將背景執行的指令]",
            f"  {argv[0]} {argv[1]} <dispatch_prompt>",
            "[dispatch_prompt 內容]",
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
