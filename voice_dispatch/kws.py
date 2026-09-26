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


def match_wake_variants(tokens, confs, prefixes, variants, min_conf: float = 0.0):
    """接受規則：轉錄 token 序列中出現 `(前綴)(變體)` 的**相鄰 bigram**，且兩個 token
    的信心度都 ≥ `min_conf`，就回傳命中的字串，否則 None。

    為什麼不比對「完全相等」：全詞彙解碼對罕見專有名詞不穩——真講 "Hermes" 常被聽成
    `homes`/`hums`，開頭的 "hey" 也可能被聽成 `a`/`the`。所以兩側都用「集合」放寬，
    而否決力來自變體集合本身：`hermit`/`mess`/`miss`/`mouse`/`mom`/`harm` 都不在裡面。
    純函式，方便單元測試。
    """
    pset = {str(w).lower().strip() for w in prefixes}
    vset = {str(w).lower().strip() for w in variants}
    for i in range(len(tokens) - 1):
        if tokens[i] in pset and tokens[i + 1] in vset:
            if min(confs[i], confs[i + 1]) >= min_conf:
                return f"{tokens[i]} {tokens[i + 1]}"
    return None


def match_wake_phrase_scored(tokens, confs, phrases):
    """同 `match_wake_phrase`，但不套門檻，改回傳 `(詞彙, 最小 token 信心度)`。

    給需要「先取候選、再自行套門檻」的呼叫者（評測工具 / 兩段式第一階）用。
    """
    for phrase in phrases:
        n = len(phrase)
        if not n or n > len(tokens):
            continue
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == tuple(phrase):
                return " ".join(phrase), min(confs[i:i + n])
    return None


def match_wake_phrase(tokens, confs, phrases, min_conf: float):
    """詞序比對：token 序列中若含某個詞彙（整段詞序相同）且每個 token 信心度
    都 ≥ `min_conf`，回傳該詞彙字串，否則 None。

    為什麼需要詞序（而不是 `word in words`）：Vosk 受限詞彙解碼**一定會吐最接近
    的詞**——即使 grammar 只有 `["hey hermes", "[unk]"]`，單喊 "hermes" 仍會被解成
    `['hermes']`（conf 1.0）。所以「要講全 hey hermes」必須在這一層用詞序擋：
    `['hermes']` 不含 `('hey','hermes')` → 不命中。

    純函式（不碰模型），方便單元測試。
    """
    m = match_wake_phrase_scored(tokens, confs, phrases)
    if m and m[1] >= min_conf:
        return m[0]
    return None


class VoskSpotter:
    """用 Vosk 的「限制詞彙解碼」當關鍵詞偵測。

    為什麼用它而不是 openWakeWord：openWakeWord 的預訓練模型不支援自訂詞
    （只有 hey_jarvis / alexa …），要「hermes」得訓練自訂模型（Colab + GB 級
    資料集）。Vosk 只要把解碼詞彙鎖成指定詞，就等於關鍵詞偵測——免訓練、
    免 AccessKey、支援任意英文詞。

    ⚠️ **詞彙表只用來「限制解碼空間」，不能拿來「拒絕」**（2026-09-26 實測）：
    把 grammar 設成 `["hey hermes", "[unk]"]` 後，只喊 "Hermes." 仍會被 Vosk
    **強制解成 `hermes`（conf 1.0）**——受限詞彙解碼一定會吐最接近的詞。
    所以「要講全 hey hermes 才算」**不能靠 grammar**，必須在 `feed()` 做
    **詞序（phrase）比對**：token 序列要真的含 `["hey","hermes"]` 才命中。

    實測（2026-09-25 / 09-26，edge-tts 英文音檔）：
      "Hey Hermes."               → tokens [hey, hermes] → 命中
      "Hey Hermes, restart…"      → [hey, hermes, [unk]] → 命中
      "Hermes."（只喊單詞）        → tokens [hermes] → **不命中**（詞序比對擋掉）
      4 段真實房間背景（共 56 秒）→ 零誤觸；單次耗時 0.05~0.08s（3 秒音檔）
    """

    def __init__(self, model_path: str, words=None, min_conf: float = 0.0, logger=None,
                 full_vocab: bool = False, use_partial: bool = False):
        import os as _os
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)                 # 別把 kaldi 的 log 灌進我們的 log
        self.words = [w.lower().strip() for w in (words or ["hermes"]) if w and w.strip()]
        self.min_conf = float(min_conf)
        # 每個詞彙切成 token 序列（"hey hermes" → ["hey","hermes"]），用於詞序比對。
        self._phrases = [tuple(p.split()) for p in self.words]
        self.full_vocab = bool(full_vocab)
        # use_partial：連「未定案的 partial 結果」也算命中＝**最低延遲**的閘門。
        # partial 沒有 per-word 信心度、且限制詞彙會硬把雜音解成詞，所以只適合當
        # 「第一階段閘門」（後面還有確認階段負責精度），不適合單獨當最終判定。
        self.use_partial = bool(use_partial)
        self._model = Model(_os.path.expanduser(model_path))
        if self.full_vocab:
            # 全詞彙解碼：**不加 grammar**。這樣才看得出實際講的是 "hey hermit"
            # 還是 "hey hermes"——限制詞彙會把近似音硬解成惟一的候選詞（見兩段式）。
            self._rec = KaldiRecognizer(self._model, 16000)
        else:
            # 限制詞彙解碼：等同關鍵詞偵測，常開路徑用（快、省）。
            self._rec = KaldiRecognizer(self._model, 16000, json.dumps([*self.words, "[unk]"]))
        self._rec.SetWords(True)        # 要 per-word conf 才擋得掉近似音誤觸
        self.latest = ""
        self.last_conf = 0.0            # 最近一次命中的最小 token 信心度（診斷用）
        if logger:
            logger.info(
                "喚醒詞引擎：vosk（%s，詞彙 %s，信心度門檻 %.2f）",
                "全詞彙" if self.full_vocab else "限制詞彙", self.words, self.min_conf)

    def feed(self, samples) -> Optional[str]:
        """餵入音訊；命中喚醒詞回傳該詞，否則 None。

        **只看 final result，不看 partial**：partial 沒有 per-word 信心度，
        在只有 1~2 個詞的限制詞彙表下很容易把雜音「強制」解成喚醒詞
        （2026-09-25 實測：啟動暫態就被解成 hermes 而誤觸）。
        代價是判定要等這句講完（約 0.3~0.6 秒），換來的是不亂觸發。

        **命中＝詞序比對**（不是單字比對）：受限詞彙解碼一定會吐最接近的詞，
        所以只喊 "hermes" 也會被解成 `hermes`——光靠 grammar 擋不掉。
        這裡要求 result 的 token 序列真的含 `["hey","hermes"]`（且每個 token 的
        conf ≥ `min_conf`）才算命中，單詞 "hermes" 因此不會觸發。
        """
        pcm = _to_pcm16(samples)
        if pcm.size == 0:
            return None
        if not self._rec.AcceptWaveform(pcm.tobytes()):
            data = json.loads(self._rec.PartialResult())
            self.latest = data.get("partial", "")
            if self.use_partial:
                # partial 沒有 per-word 信心度 → 只做詞序比對，不套 conf。
                # （精度交給下游確認階段；這裡的目標是「最早觸發」。）
                toks = self.latest.split()
                m = match_wake_phrase_scored(toks, [1.0] * len(toks), self._phrases)
                if m:
                    self.last_conf = 1.0
                    return m[0]
            return None
        data = json.loads(self._rec.Result())
        self.latest = data.get("text", "")
        result = data.get("result") or []
        tokens = [str(w.get("word", "")).lower() for w in result]
        confs = [float(w.get("conf", 0.0)) for w in result]
        m = match_wake_phrase_scored(tokens, confs, self._phrases)
        if m and m[1] >= self.min_conf:
            self.last_conf = m[1]
            return m[0]
        return None

    def verify_utterance(self, samples) -> Optional[str]:
        """整段（非串流）辨識後做詞序比對——**兩段式的第二階段確認**用。

        與 `feed()` 共用同一顆模型與詞彙表，但一次吃完整段音訊（`Reset` →
        全部餵入 → `FinalResult`）。第二階段只在高精度模型上跑，且僅在第一階段
        出現候選時才呼叫，所以「常開路徑不跑重模型」的原則不受影響。
        """
        pcm = _to_pcm16(samples)
        self._rec.Reset()
        if pcm.size:
            self._rec.AcceptWaveform(pcm.tobytes())
        data = json.loads(self._rec.FinalResult())
        self.latest = data.get("text", "")
        result = data.get("result") or []
        tokens = [str(w.get("word", "")).lower() for w in result]
        confs = [float(w.get("conf", 0.0)) for w in result]
        m = match_wake_phrase_scored(tokens, confs, self._phrases)
        if m and m[1] >= self.min_conf:
            self.last_conf = m[1]
            return m[0]
        return None


class WakeVerifier:
    """兩段式的**第二階段**：對第一階段的候選做「真的是喚醒詞嗎」的確認。

    為什麼要它（2026-09-26 實測結論）：第一階段用**限制詞彙**解碼當快速關鍵詞偵測
    （省、快，但**一定會把最接近的近似音硬解成喚醒詞**——"hey hermit"、"hay her mess"
    都被解成 `hey hermes` 且 conf=1.0，所以光靠 confidence 擋不掉，實測誤判 ~8.5%）。
    第二階段改用**同顆（或更大）模型的「全詞彙」解碼**，看它實際轉出什麼字：
    "hey hermit"→`hey hermit`、"hay her mess"→`hey her mess`，就否決掉了。

    但全詞彙解碼對罕見專有名詞不穩（"Hermes" 常被聽成 `homes`/`hums`），所以接受
    規則不是完全相等，而是 **`(hey 類詞)(hermes 類詞)` 的相鄰 bigram**（可設定）。
    實測（664 句近似發音語料）誤判 0/600、漏判 ~8%，且仍用同一顆 small 模型 →
    零額外記憶體、每次確認 ~50ms（延遲可控）。
    """

    def __init__(self, model_path: str, prefixes, variants, min_conf: float = 0.0, logger=None):
        import os as _os
        from vosk import KaldiRecognizer, Model, SetLogLevel

        SetLogLevel(-1)
        self.prefixes = {str(w).lower().strip() for w in prefixes if str(w).strip()}
        self.variants = {str(w).lower().strip() for w in variants if str(w).strip()}
        self.min_conf = float(min_conf)
        self._model = Model(_os.path.expanduser(model_path))
        self._rec = KaldiRecognizer(self._model, 16000)   # 全詞彙（不加 grammar）
        self._rec.SetWords(True)
        self.latest = ""
        if logger:
            logger.info(
                "喚醒確認器：vosk 全詞彙（前綴 %s / 變體 %s，信心度門檻 %.2f）",
                sorted(self.prefixes), sorted(self.variants), self.min_conf)

    def verify(self, samples) -> bool:
        """整段辨識後套 bigram 規則；命中回 True。清空辨識器狀態，可重複呼叫。"""
        pcm = _to_pcm16(samples)
        self._rec.Reset()
        if pcm.size:
            self._rec.AcceptWaveform(pcm.tobytes())
        data = json.loads(self._rec.FinalResult())
        self.latest = data.get("text", "")
        result = data.get("result") or []
        tokens = [str(w.get("word", "")).lower() for w in result]
        confs = [float(w.get("conf", 0.0)) for w in result]
        return match_wake_variants(tokens, confs, self.prefixes, self.variants,
                                   self.min_conf) is not None


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
