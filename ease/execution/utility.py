"""UtilityEvaluator —— 检索结果效用过滤（构想三过滤层）。

score = α·relevance + β·novelty − γ·redundancy
- relevance：与当前缺口的嵌入余弦
- redundancy：与已有证据的最大余弦；>dup_threshold 直接丢弃（同源无增量）
- novelty：新 token 占比（对已有证据集合）
- 会话级证据指纹缓存（memory.fingerprints）防重复检索/重复入库
全部基于真实嵌入计算。
"""
import re

from .memory import WorkingMemory


class UtilityEvaluator:
    _STOPWORDS = set(
        "the a an of and or to in on at for from by with as is are was were be been being "
        "this that these those it its he she they we you i me my your".split()
    )

    def __init__(self, embedder, config=None):
        self.embedder = embedder
        cfg = config or {}
        self.relevance_w = cfg.get("relevance_w", 0.5)
        self.novelty_w = cfg.get("novelty_w", 0.3)
        self.redundancy_w = cfg.get("redundancy_w", 0.2)
        self.keep_threshold = cfg.get("keep_threshold", 0.35)
        self.dup_threshold = cfg.get("redundancy_dup_threshold", 0.95)

    def filter(self, docs, gap_text, memory):
        """返回 (kept, dropped)；dropped 为 [(doc, reason)]。"""
        if not docs:
            return [], []
        qv = self.embedder.embed(gap_text)[0]
        existing_texts = [e.raw_text for e in memory.evidence]
        existing_embs = self.embedder.embed(existing_texts) if existing_texts else None
        existing_tokens = set()
        for e in memory.evidence:
            existing_tokens |= self._tokens(e.raw_text)

        doc_embs = self.embedder.embed([d.text for d in docs])
        qsims = doc_embs @ qv

        kept, dropped = [], []
        for i, doc in enumerate(docs):
            rel = float(qsims[i])
            fp = self.fingerprint(doc.text)
            if memory.seen(fp):
                dropped.append((doc, "duplicate_fingerprint"))
                continue
            red = 0.0
            if existing_embs is not None:
                sims = existing_embs @ doc_embs[i]
                red = float(sims.max()) if sims.size else 0.0
            if red > self.dup_threshold:
                dropped.append((doc, f"redundant cos={red:.2f}"))
                continue
            toks = self._tokens(doc.text)
            new_toks = toks - existing_tokens
            novelty = len(new_toks) / (len(toks) or 1)
            score = self.relevance_w * rel + self.novelty_w * novelty - self.redundancy_w * red
            if score >= self.keep_threshold:
                kept.append((doc, score))
            else:
                dropped.append((doc, f"low_score {score:.2f}"))
        return kept, dropped

    @staticmethod
    def fingerprint(text):
        toks = sorted(UtilityEvaluator._tokens(text))
        return " ".join(toks[:200])

    @staticmethod
    def _tokens(text):
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return {t for t in toks if t not in UtilityEvaluator._STOPWORDS and len(t) > 2}
