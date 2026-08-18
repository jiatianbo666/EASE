"""诊断：EASE 窄查询 vs RAG-Once 原始问题，到底差在哪。

对 Dutch 题（5abc197055）跑一次真实 EASE，dump 出：
  - 子问题分解
  - 每轮实际查询 + 检索返回的前 k 个文档（是否含 gold、分数）
  - utility filter 的 keep/drop 及原因
  - 压缩后 memory 里的事实（是否含答案 Dutch / Carice van Houten）
再把 RAG-Once 的原始问题跑同一 search tool（k=8），对比 top-k。
"""
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config
from ease.llm.client import LLMClient, reset_stats
from ease.embeddings.embedder import Embedder
from ease.search.offline import OfflineSearchTool
from ease.experience.skill_store import SkillStore
from ease.agent.ease_agent import EASEAgent
from ease.eval.data_loader import sample_questions

QID = sys.argv[1] if len(sys.argv) > 1 else "5abc197055"   # 默认 Dutch heritage 题
cfg = load_config()
reset_stats()
embedder = Embedder()
llm = LLMClient(cfg["llm"])
search = OfflineSearchTool(cfg["search"]["offline"], embedder=embedder)

qs = sample_questions(10, seed=42)
q = next(x for x in qs if x["_id"].startswith(QID))
print(f"问题: {q['question']}")
print(f"gold: {q['answer']!r}\n")

# ---- RAG-Once 对照组：原始问题直接检索 ----
print("=" * 70)
print("对照 A: RAG-Once 原始问题检索 (evidence_search k=8)")
print("=" * 70)
rag_docs = search.evidence_search(q["question"], k=8, qid=q["_id"])
for d in rag_docs:
    tag = "GOLD*" if d.is_gold else "     "
    print(f"  {tag} #{d.rank:>2} score={d.score:.3f} {d.title[:60]}")
    if d.is_gold:
        idx = d.text.lower().find("dutch")
        print(f"        ...{'Dutch 命中于: ...' + d.text[max(0,idx-80):idx+80].replace(chr(10),' ') if idx>=0 else 'text 无 Dutch 字样'}")
gold_ids = [d.doc_id for d in rag_docs if d.is_gold]
print(f"\n  → 原始问题 top-8 含 {len(gold_ids)} 个 gold 证据文档: {gold_ids}")

# ---- EASE 真实跑 ----
print("\n" + "=" * 70)
print("对照 B: EASE 全流程真实跑（空技能库）")
print("=" * 70)
import tempfile
_tmp = os.path.join(tempfile.mkdtemp(), "diag.db")
store = SkillStore({**cfg["experience"], "skills_db": _tmp}, embedder=embedder)
agent = EASEAgent(cfg, llm, search, store, embedder=embedder)
result = agent.run(q["question"], qid=q["_id"])
t = result.trace

print(f"\n[子问题分解]")
for s in t.sub_questions:
    print(f"  [{s['qtype']}] {s['text']}")

print(f"\n[每轮查询 → 检索返回]")
for i, rec in enumerate(t.searches, 1):
    print(f"\n  --- search #{i} ---")
    print(f"  query : {rec['query']!r}")
    print(f"  focus : {rec['focus']!r}")
    # 从 search.call_log 找这次调用的 gold_hits（按查询匹配）
    for cl in search.call_log:
        if cl.get("query") == rec["query"]:
            print(f"  retr  : n={cl['n']} gold_hits={cl.get('gold_hits', '?')}")
            break
    for d in rec["kept"]:
        print(f"    KEEP {d[:60]}")
    for d, why in rec["dropped"]:
        print(f"    DROP {d[:50]}  <- {why}")

print(f"\n[memory 最终证据事实]")
for e in result.memory.evidence:
    print(f"  [{e.eid}] {e.title[:55]}")
    for f in e.facts:
        if f.predicate:
            print(f"      {f.subject} — {f.predicate}: {f.value}")
        else:
            print(f"      {f.subject}: {f.value}")

print(f"\n[结果] answer={result.answer!r}  EM={result.trace.answer==q['answer']}")
print(f"  覆盖度历史: {t.coverage_hist}")
print(f"  searches={t.budget_used_searches} llm_calls={t.llm_calls} cost=${t.cost_usd:.4f}")
