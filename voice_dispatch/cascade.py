"""串接式喚醒：第一階段「hey」閘門 → 第二階段「hermes」確認。

動機（2026-09-26 使用者定案「極限精準 + 極限延遲」）：

- **常開路徑越省越好** → 第一階段只認「hey」一個字（Vosk 限制詞彙），且可用
  **partial（未定案結果）** 即時觸發 = 最早醒、延遲最低。
- **精度靠第二階段** → 觸發後才對一段音訊做**全詞彙**解碼，用
  `(前綴)(hermes 變體)` bigram 規則確認。限制詞彙會把 "hermit"/"her mess" 硬解成
  hermes（實測誤判 ~8.5%），全詞彙才分得出來（實測誤判 0/600）。

為什麼這樣切最省：第一階段天生誤判高但**不漏**（只當閘門）；真正花力氣的解碼只在
閘門觸發後、對一小段音訊跑，且**一確認就立刻喚醒**（不必等句子結束的靜音判定）。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from . import kws

SR = 16000


class WakeCascade:
    """吃音訊區塊、回報是否確認喚醒。與 I/O 無關，方便單元測試與離線評測。"""

    def __init__(self, cfg, logger=None):
        w = cfg.wake
        self._log = logger
        self.block = max(1, int(cfg.audio.blocksize))
        # 第一階段：hey 閘門（限制詞彙、可 partial）
        self.gate = kws.VoskSpotter(
            w.vosk_model, words=w.vosk_words, min_conf=w.vosk_min_conf,
            use_partial=getattr(w, "gate_partial", True), logger=logger,
        )
        # 第二階段：hermes 確認（受限於 openwakeword 引擎不適用）
        self.verifier = None
        if getattr(w, "verify_enabled", False) and w.kws_engine == "vosk":
            self.verifier = kws.WakeVerifier(
                w.verify_model or w.vosk_model,
                prefixes=w.verify_prefixes, variants=w.verify_variants,
                min_conf=w.verify_min_conf, logger=logger,
            )
        self.preroll_blocks = max(1, int(round(w.preroll_sec * SR / self.block)))
        self.window_blocks = max(1, int(round(w.verify_window_sec * SR / self.block)))
        self.interval_blocks = max(1, int(round(w.verify_interval_sec * SR / self.block)))
        self.reset()

    def reset(self, hard: bool = False) -> None:
        """回到監聽狀態。

        hard=True 會**連內部辨識器狀態一起清空**——用在「把每段獨立音訊當全新一次」
        的場合（離線評測）。常駐 daemon 是連續串流、不需要清，故預設 False（清掉
        辨識器反而會丟掉正在緩衝的音訊）。
        """
        self._ring = deque(maxlen=self.preroll_blocks)
        self._armed = None          # None＝監聽中；list＝已觸發、正在確認
        self._since = 0             # 觸發後已收幾個 block
        self._next_eval = 0
        if hard:
            # 用「重建辨識器」而非 `_rec.Reset()`——實測 Reset() 清不乾淨狀態。
            self.gate.reset_recognizer()
            if self.verifier is not None:
                self.verifier.reset_recognizer()

    def feed(self, block) -> Optional[str]:
        """餵一個 block；確認喚醒回傳描述字串，否則 None。"""
        if self._armed is None:
            self._ring.append(block)
            if self.gate.feed(block):
                # 閘門命中：把 preroll 一起帶進確認窗（確保含完整「hey」）
                self._armed = list(self._ring)
                self._since = 0
                self._next_eval = self.interval_blocks
            return None

        # 已觸發：累積音訊，週期性確認
        self._armed.append(block)
        self._since += 1

        if self.verifier is None:
            self.reset()            # 無確認階段 → 閘門命中即喚醒
            return "hey（閘門命中，無確認階段）"

        if self._since >= self._next_eval:
            if self.verifier.verify(np.concatenate(self._armed)):
                desc = f"hey hermes（確認轉錄 {self.verifier.latest!r}）"
                self.reset()
                return desc
            self._next_eval += self.interval_blocks

        if self._since >= self.window_blocks:
            if self._log:
                # 附上 (token, conf)：文字對但 conf 低 → 是門檻問題；文字不對 →
                # 是變體集合/acoustic 問題。（2026-09-26 加，供 recall 調校診斷）
                scored = getattr(self.verifier, "latest_scored", None)
                self._log.info("hey 閘門觸發但窗內未見 hermes（轉錄 %r, tokens=%s）→ 忽略",
                               self.verifier.latest, scored)
            self.reset()
        return None
