"""神經網路喚醒詞（openWakeWord）與神經網路 VAD（Silero）——真實語音助手的做法。

為什麼要換掉舊做法
------------------
舊做法是「拍手兩下 → 錄 2.5 秒 → 對那 2.5 秒做 STT → 比對字串裡有沒有 'hermes'」。
三個結構性缺點（2026-09-25 使用者實測反饋「太智障」）：

1. 慢——命中後還要 2.5s 錄音 + 2.6s STT 才開始回應。
   Home Assistant 官方文件講得很直白：
   「The wake words have to be processed extremely fast: You can't have a voice
    assistant start listening 5 seconds after a wake word is spoken.」
2. 脆——STT 對「短音訊 + 前後靜音」極易吐幻覺（實測回吐過
   「感謝收看。」「那個更難,那個更難,那個更難。」）。
3. 還要先拍手——完全不像在跟人講話。

真實助手（Alexa / Google / Siri / Home Assistant）都不對常開路徑做 STT，
而是跑一顆**微小的關鍵詞模型（KWS）**，每 80ms 推論一次，命中才把音訊送去 STT。

本模組提供兩個元件：
  WakeWordSpotter —— openWakeWord 包裝，每 80ms 判一次喚醒詞（延遲 < 0.1s）
  SileroVad       —— Silero 神經網路 VAD，取代「RMS 門檻」判斷有沒有人講話

兩者都吃 1280 samples（80ms @16kHz）的單位；本模組內建重新切塊，
所以呼叫端可以直接餵 daemon 的 1024-sample 區塊。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

CHUNK = 1280          # 80ms @16kHz：openWakeWord 與 Silero VAD 共同的推論單位


class _Chunker:
    """把任意長度的 float32 區塊累積並重新切成固定大小的區塊。"""

    def __init__(self, size: int = CHUNK):
        self.size = size
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, samples) -> List[np.ndarray]:
        a = np.asarray(samples, dtype=np.float32).ravel()
        if a.size:
            self._buf = a if self._buf.size == 0 else np.concatenate([self._buf, a])
        out: List[np.ndarray] = []
        while self._buf.size >= self.size:
            out.append(self._buf[: self.size])
            self._buf = self._buf[self.size:]
        return out


def _to_pcm16(chunk: np.ndarray) -> np.ndarray:
    """float32 [-1,1] → int16。NaN/爆音先夾住，避免模型吃到垃圾。"""
    safe = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
    return (np.clip(safe, -1.0, 1.0) * 32767.0).astype(np.int16)


class WakeWordSpotter:
    """openWakeWord 喚醒詞偵測。不用拍手、不對喚醒詞做 STT。"""

    def __init__(self, models: List[str], threshold: float = 0.5, logger=None):
        from openwakeword.model import Model   # 延遲載入（吃 onnxruntime）

        self._model = Model(
            wakeword_models=list(models), inference_framework="onnx"
        )
        self.models = list(self._model.models.keys())
        self.threshold = float(threshold)
        self.latest: Dict[str, float] = {name: 0.0 for name in self.models}
        self.peak: Dict[str, float] = {name: 0.0 for name in self.models}
        self._chunker = _Chunker()
        if logger:
            logger.info(
                "喚醒詞引擎：openWakeWord %s（門檻 %.2f，每 %.0fms 判一次）",
                self.models, self.threshold, CHUNK / 16000 * 1000,
            )

    def feed(self, samples) -> Optional[str]:
        """餵入音訊區塊；命中喚醒詞回傳模型名，否則 None。"""
        hit: Optional[str] = None
        for chunk in self._chunker.push(samples):
            for name, score in self._model.predict(_to_pcm16(chunk)).items():
                s = float(score)
                self.latest[name] = s
                if s > self.peak[name]:
                    self.peak[name] = s
                if s >= self.threshold and hit is None:
                    hit = name
        return hit


class VoskSpotter:
    """用 Vosk 的「限制詞彙解碼」當關鍵詞偵測。

    為什麼用它而不是 openWakeWord：openWakeWord 的預訓練模型不支援自訂詞
    （只有 hey_jarvis / alexa …），要「hermes」得訓練自訂模型（Colab + GB 級
    資料集）。Vosk 只要把解碼詞彙鎖成 `["hermes", "[unk]"]`，就等於關鍵詞
    偵測——免訓練、免 AccessKey、支援任意英文詞。

    實測（2026-09-25）：
      "Hermes" / "Hey Hermes, restart the server" → 命中
      "Good morning everyone" / "The weather is nice today" → 不命中
      喇叭→手機麥克風播 "Hey Hermes" → 命中
      4 段真實房間背景（共 56 秒）→ 零誤觸；單次耗時 0.05~0.08s（3 秒音檔）
    """

    def __init__(self, model_path: str, words=None, min_conf: float = 0.0, logger=None):
        import os as _os
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)                 # 別把 kaldi 的 log 灌進我們的 log
        self.words = [w.lower() for w in (words or ["hermes"])]
        self.min_conf = float(min_conf)
        self._model = Model(_os.path.expanduser(model_path))
        self._rec = KaldiRecognizer(self._model, 16000, json.dumps([*self.words, "[unk]"]))
        self._rec.SetWords(True)        # 要 per-word conf 才擋得掉近似音誤觸
        self.latest = ""
        if logger:
            logger.info("喚醒詞引擎：vosk（詞彙 %s，信心度門檻 %.2f）", self.words, self.min_conf)

    def feed(self, samples) -> Optional[str]:
        """餵入音訊；命中喚醒詞回傳該詞，否則 None。

        **只看 final result，不看 partial**：partial 沒有 per-word 信心度，
        在只有 1~2 個詞的限制詞彙表下很容易把雜音「強制」解成喚醒詞
        （2026-09-25 實測：啟動暫態就被解成 hermes 而誤觸）。
        代價是判定要等這句講完（約 0.3~0.6 秒），換來的是不亂觸發。
        """
        pcm = _to_pcm16(samples)
        if pcm.size == 0:
            return None
        if not self._rec.AcceptWaveform(pcm.tobytes()):
            self.latest = json.loads(self._rec.PartialResult()).get("partial", "")
            return None
        data = json.loads(self._rec.Result())
        self.latest = data.get("text", "")
        for w in data.get("result") or []:
            word = str(w.get("word", "")).lower()
            if word in self.words and float(w.get("conf", 0.0)) >= self.min_conf:
                return word
        return None


class SileroVad:
    """Silero 神經網路 VAD：判斷「這 80ms 有沒有人在講話」。

    比 RMS 門檻穩健得多——RMS 門檻要嘛太敏感（冷氣／風扇被判成語音），
    要嘛太保守（隔著桌面講話被判成沒人講話，程式就一直重問需求）。
    """

    FRAME = 480           # Silero 內部推論單位

    def __init__(self, threshold: float = 0.5, model_path: str = "",
                 rms_gate: float = 0.0, logger=None):
        import onnxruntime as ort

        path = os.path.expanduser(
            model_path or "~/.local/share/hermes-voice-dispatch/silero_vad.onnx"
        )
        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.threshold = float(threshold)
        self.rms_gate = float(rms_gate)
        self.last_prob = 0.0
        self._sr = np.array(16000, dtype=np.int64)
        self._chunker = _Chunker()
        self._reset_states()
        if logger:
            logger.info("端點偵測：Silero VAD（門檻 %.2f，RMS 閘 %.4f）",
                        self.threshold, self.rms_gate)

    def _reset_states(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _prob(self, chunk) -> float:
        x = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
        probs = []
        for i in range(0, max(0, x.size - self.FRAME + 1), self.FRAME):
            frame = x[i:i + self.FRAME].reshape(1, -1)
            out = self._sess.run(None, {
                "input": frame, "sr": self._sr, "h": self._h, "c": self._c,
            })
            self._h, self._c = out[1], out[2]
            probs.append(float(out[0][0][0]))
        return float(np.mean(probs)) if probs else 0.0

    def feed(self, samples) -> Optional[bool]:
        """餵入音訊區塊；有完整 80ms 區塊時回傳該區塊是否為語音，否則 None。"""
        verdict: Optional[bool] = None
        for chunk in self._chunker.push(samples):
            rms = float(np.sqrt(np.mean(np.square(
                np.asarray(chunk, dtype=np.float64)))))
            if self.rms_gate > 0.0 and rms < self.rms_gate:
                self.last_prob = 0.0      # 安靜 → 跳過推論（省 CPU）
                verdict = False
                continue
            try:
                self.last_prob = self._prob(chunk)
            except Exception:  # noqa: BLE001 —— VAD 壞掉不該讓整個守護程式掛掉
                continue
            verdict = self.last_prob >= self.threshold
        return verdict
