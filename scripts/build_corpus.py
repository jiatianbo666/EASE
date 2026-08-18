"""M2 —— 从 HotpotQA 构建离线检索语料 + 稠密索引（真实计算）。

步骤：
  1. 读取 data/raw/hotpot_dev_distractor_v1.json（7405 题）
  2. 构建去重语料 docs.jsonl（含 gold_for 映射）
  3. bge-small 嵌入全部段落 → embeddings.npy
  4. faiss IndexFlatIP → faiss.index
打印真实统计（段数/标题去重/gold 覆盖）。
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ease.utils.config import load_config, PROJECT_ROOT
from ease.search.corpus import build_docs, save_docs, load_docs
from ease.embeddings.embedder import Embedder


def main():
    cfg = load_config()
    corpus_dir = PROJECT_ROOT / cfg["search"]["offline"]["corpus_dir"]
    corpus_dir.mkdir(parents=True, exist_ok=True)

    raw_path = PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"原始数据缺失：{raw_path}")

    # ---- 1/2. 构建语料 ----
    print("读取原始数据 ...")
    import json
    with open(raw_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  问题总数: {len(questions)}")

    t0 = time.time()
    docs = build_docs(questions, source_file=os.path.basename(raw_path))
    docs_path = corpus_dir / "docs.jsonl"
    save_docs(docs, docs_path)
    print(f"  语料段落: {len(docs)}（title 去重后）  保存: {docs_path}  ({time.time()-t0:.1f}s)")

    # 真实统计
    n_gold_mapped = sum(1 for d in docs if d.gold_for)
    n_questions_covered = len({qid for d in docs for qid in d.gold_for})
    print(f"  含 gold_for 的段落: {n_gold_mapped}")
    print(f"  有任意 gold 证据被语料覆盖的问题: {n_questions_covered}/{len(questions)}")

    # ---- 3. 嵌入 ----
    print("嵌入全部段落（bge-small-en-v1.5, cuda）...")
    emb = Embedder(device="cuda")
    print(f"  模型加载: {emb.load_seconds:.1f}s, dim={emb.dim}")
    texts = [d.text for d in docs]
    vecs = emb.embed(texts, batch_size=128, progress=True)
    emb_path = corpus_dir / "embeddings.npy"
    np.save(emb_path, vecs)
    print(f"  嵌入保存: {emb_path}  shape={vecs.shape}")

    # ---- 4. faiss 索引 ----
    import faiss
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    idx_path = corpus_dir / "faiss.index"
    faiss.write_index(index, str(idx_path))
    print(f"  faiss 索引保存: {idx_path}  nvec={index.ntotal}")

    print("=" * 50)
    print("✅ 语料构建完成。真实验证下一步：对真实问题检索 top-5，核对 gold 命中。")


if __name__ == "__main__":
    main()
