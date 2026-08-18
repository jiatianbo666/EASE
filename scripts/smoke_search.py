"""M2 真实验证 —— 离线检索对真实问题检索 top-5，核对 gold 证据命中。

从 HotpotQA 取 3 条真实问题，用原始问题作为查询检索，打印结果，
检查 gold 标题是否出现在检索结果中（离线工具按 doc.gold_for 标注 is_gold）。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config, PROJECT_ROOT
from ease.embeddings.embedder import Embedder
from ease.search.offline import OfflineSearchTool

SAMPLE_QIDS = [
    "5a8b57f25542995d1e6f1371",  # Were Scott Derrickson and Ed Wood of the same nationality?
]


def main():
    cfg = load_config()
    with open(PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json", encoding="utf-8") as f:
        questions = json.load(f)
    by_id = {q["_id"]: q for q in questions}

    tool = OfflineSearchTool(cfg["search"]["offline"], embedder=Embedder())
    print(f"语料加载完成: {len(tool.docs)} 段落, calls={tool.calls}")
    print("=" * 60)

    picked = [by_id[qid] for qid in SAMPLE_QIDS if qid in by_id]
    if not picked:
        picked = questions[:3]

    total_gold_top5 = 0
    for q in picked:
        qid = q["_id"]
        gold_titles = {t for t, _ in q["supporting_facts"]}
        print(f"\nQ: {q['question']}")
        print(f"  gold titles: {sorted(gold_titles)}")
        res = tool.evidence_search(q["question"], k=5, qid=qid)
        hits = [r for r in res if r.is_gold]
        total_gold_top5 += len(hits)
        for r in res:
            mark = "★GOLD" if r.is_gold else "     "
            print(f"  {mark} {r.rank}. [{r.score:.3f}] {r.title}")
            print(f"        {r.snippet[:110]}")

    # adaptive-k 检查
    print("\n" + "=" * 60)
    print("adaptive-k 检查（evidence_search 动态 k）：")
    for q in picked:
        res = tool.evidence_search(q["question"], gap_slot=None, qid=q["_id"])
        print(f"  k={len(res)}  查询: {q['question'][:50]}...")
    print(f"\ncalls 总数: {tool.calls}（真实计数）")

    print("\n✅ M2 检索冒烟完成（gold 命中情况见上；top5 gold 命中 {}/{}）".format(
        total_gold_top5, len(picked)))


if __name__ == "__main__":
    main()
