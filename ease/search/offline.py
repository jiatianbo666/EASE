"""离线语料检索后端 —— BM25 + 稠密混合检索（真实索引）。

索引构建：scripts/build_corpus.py（docs.jsonl + embeddings.npy + faiss.index）
本类负责加载索引并提供 SearchTool 原子工具：
  - 候选集：faiss 稠密 top-K（K=50，归一化向量内积 ≈ 余弦）
  - 混合打分：score = λ·bm25_norm + (1-λ)·dense_cos
  - adaptive-k：evidence_search 按分数曲线动态截断，降低返回 token
"""
import os
import time

import numpy as np

from .base import SearchTool, RetrievedDoc
from .corpus import load_docs
from ..utils.config import PROJECT_ROOT


class OfflineSearchTool(SearchTool):
    def __init__(self, config, embedder=None):
        super().__init__()
        self.corpus_dir = PROJECT_ROOT / config.get("corpus_dir", "data/corpus")
        self.lmb = config.get("lambda_bm25", 0.3)
        self.fast_k = config.get("fast_lookup_k", 2)
        self.dense_candidates = config.get("dense_candidates", 50)
        # 候选并集：dense top-K ∪ bm25 top-K（K 可独立配置）。
        # 单一 dense 门会剪掉"精确匹配但嵌入漂移"的文档（人名+抽象属性查询），
        # 并集保住 BM25 的精确信号。
        self.bm25_candidates = config.get("bm25_candidates", 50)
        if embedder is None:
            from ..embeddings.embedder import Embedder
            embedder = Embedder()
        self.embedder = embedder

        docs_path = self.corpus_dir / "docs.jsonl"
        if not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"语料不存在：{docs_path}\n请先运行 scripts/build_corpus.py"
            )
        self.docs = load_docs(docs_path)
        self.texts = [d.text for d in self.docs]
        self.titles = [d.title for d in self.docs]
        self.by_id = {d.doc_id: d for d in self.docs}

        from rank_bm25 import BM25Okapi
        # 小写归一：BM25 大小写敏感，query 大写/小写漂移会整个失配
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])

        import faiss
        emb_path = self.corpus_dir / "embeddings.npy"
        idx_path = self.corpus_dir / "faiss.index"
        if not (os.path.exists(emb_path) and os.path.exists(idx_path)):
            raise FileNotFoundError(
                f"稠密索引缺失：{emb_path} / {idx_path}\n请先运行 scripts/build_corpus.py"
            )
        self.emb = np.load(emb_path)
        self.index = faiss.read_index(str(idx_path))

    # ---------- 工具实现 ----------
    def fast_lookup(self, query, k=2, qid=None):
        return self._retrieve(query, k=k, adaptive=False, qid=qid)

    def evidence_search(self, query, gap_slot=None, k=None, qid=None):
        return self._retrieve(query, k=k, adaptive=True, qid=qid)

    def deep_scrape(self, doc_ref):
        """doc_ref 为 doc_id；返回全文 Doc。"""
        d = self.by_id.get(doc_ref)
        self._count(doc_ref, [d] if d else [], 0.0)
        return d

    # ---------- 内部 ----------
    def _retrieve(self, query, k=None, adaptive=False, qid=None):
        t0 = time.time()
        qv = self.embedder.embed(query)[0]
        K = self.dense_candidates
        dense_scores, idxs = self.index.search(np.asarray([qv], dtype="float32"), K)
        dense_scores = dense_scores[0]
        cand_idx = idxs[0]
        valid = cand_idx >= 0
        dense_cand = cand_idx[valid]

        # 候选并集：dense top-K ∪ bm25 top-K。
        # 单一 dense 门会在嵌入漂移时（如人名+抽象属性查询）把 BM25 精确匹配的
        # 文档剪在候选之外；并集让 BM25 信号参与重排。
        bm_all = self.bm25.get_scores(query.lower().split())
        bm_ids = np.argsort(-bm_all)[:self.bm25_candidates]
        union = np.unique(np.concatenate([dense_cand, bm_ids])).astype(int)

        bm_c = bm_all[union]
        bm_max = bm_c.max()
        bm_norm = bm_c / bm_max if bm_max > 0 else bm_c
        # 向量已归一化，内积=余弦。dense 同按候选 max 归一到 [0,1]：
        # 两个信号对称可比，λ 权重才有真实含义；否则并集引入的 BM25 重长文档
        # 会撑大分母、压扁短 gold 文档的 BM25 分量（实测回归）。
        dense_u = self.emb[union] @ qv
        dense_max = dense_u.max()
        dense_norm = dense_u / dense_max if dense_max > 0 else dense_u

        hybrid = self.lmb * bm_norm + (1.0 - self.lmb) * dense_norm
        order = np.argsort(-hybrid)
        cand_idx = union[order]
        hybrid_sorted = hybrid[order]

        if adaptive and k is None:
            k = self._adaptive_k(hybrid_sorted)
        elif k is None:
            k = self.fast_k
        k = max(1, min(k, len(cand_idx)))

        results = []
        for rank, (doc_i, score) in enumerate(zip(cand_idx[:k], hybrid_sorted[:k])):
            doc_i = int(doc_i)
            d = self.docs[doc_i]
            is_gold = False
            if qid:
                g = d.gold_for.get(qid, [])
                is_gold = len(g) > 0
            results.append(RetrievedDoc(
                doc_id=d.doc_id, title=d.title, text=d.text,
                snippet=d.text[:200], score=float(score), rank=rank + 1,
                is_gold=is_gold,
            ))
        gold_hits = sum(1 for r in results if r.is_gold)
        self._count(query, results, (time.time() - t0) * 1000, gold_hits=gold_hits)
        return results

    def _adaptive_k(self, sorted_scores):
        """按分数曲线找截断点（降 token 25-75%）：
        从第 2 项起，分数跌破 max*0.5 或相对上一项骤降(0.75)即截断；clamp 到 [1,5]。
        """
        n = len(sorted_scores)
        if n == 0:
            return 0
        mx = float(sorted_scores[0])
        if mx <= 0:
            return min(3, n)
        k = 1
        for i in range(1, n):
            s = float(sorted_scores[i])
            if s < 0.5 * mx:
                break
            if float(sorted_scores[i - 1]) > 0 and s < 0.75 * float(sorted_scores[i - 1]):
                break
            k = i + 1
        return max(1, min(k, 5))
