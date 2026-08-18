"""修复验证 —— 对曾失败题跑 EASE，确认查询已为英文且能正确检索/作答（全真实）。"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config, PROJECT_ROOT
from ease.llm.client import LLMClient
from ease.embeddings.embedder import Embedder
from ease.search.offline import OfflineSearchTool
from ease.experience.skill_store import SkillStore
from ease.agent.ease_agent import EASEAgent
from ease.eval.metrics import em_f1

QID = "5ae143ed55429920d5234360"  # 曾失败：中文槽文本→检索空


def main():
    cfg = load_config()
    embedder = Embedder()
    llm = LLMClient(cfg["llm"])
    store = SkillStore(cfg["experience"], embedder=embedder)
    search = OfflineSearchTool(cfg["search"]["offline"], embedder=embedder)
    agent = EASEAgent(cfg, llm, search, store, embedder=embedder)

    with open(PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json", encoding="utf-8") as f:
        questions = json.load(f)
    q = next(x for x in questions if x["_id"] == QID)

    print(f"Q: {q['question']}")
    print(f"gold: {q['answer']!r}")
    result = agent.run(q["question"], qid=q["_id"])
    t = result.trace

    print(f"\n=== {len(t.searches)} 次检索（查询是否英文？）===")
    non_ascii = 0
    for s in t.searches:
        q_ascii = all(ord(c) < 128 for c in s["query"])
        if not q_ascii:
            non_ascii += 1
        print(f"  {s['query'][:70]!r}  kept={len(s['kept'])}  focus={s['focus'][:30]!r}")

    em, f1 = em_f1(result.answer, q.get("answer", ""))
    print(f"\n答案: {result.answer[:200]}")
    print(f"EM={em} F1={f1}  searches={t.budget_used_searches}  cost=${t.cost_usd:.4f}")
    print(f"覆盖度: {t.coverage_hist}")

    ok = (non_ascii == 0 and em > 0)
    print(f"\n{'✅' if ok else '❌'} 修复验证{'通过' if ok else '未通过'} "
          f"(非英文查询 {non_ascii} 条, EM={em})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
