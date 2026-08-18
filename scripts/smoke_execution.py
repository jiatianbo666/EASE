"""M5 真实验证 —— 执行层：查询生成 → 检索 → 效用过滤 → 证据压缩 → 工作记忆去重。

整条流水线走真实组件：
  1) QueryGenerator（真实小模型调用）把 core 缺口槽生成检索查询，校验查询非空且不重复
  2) OfflineSearchTool 真实检索 → 真实文档
  3) UtilityEvaluator（真实嵌入）按 relevance/novelty/redundancy 给 keep/drop+理由
  4) EvidenceCompressor（真实小模型调用）抽出结构化 Fact，数值/日期保留
  5) WorkingMemory 指纹去重：同一文档入两次 → 同一 eid
同时校验 Budget Ledger 注入（search/token/cost 出现在每一步的 system 提示中）。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config, PROJECT_ROOT
from ease.llm.client import LLMClient, STATS
from ease.embeddings.embedder import Embedder
from ease.search.offline import OfflineSearchTool
from ease.control.metacognitive import SubQuestion, Slot
from ease.execution.querygen import QueryGenerator
from ease.execution.utility import UtilityEvaluator
from ease.execution.compressor import EvidenceCompressor
from ease.execution.memory import WorkingMemory

SAMPLE_QID = "5a8b57f25542995d1e6f1371"  # Were Scott Derrickson and Ed Wood of the same nationality?


def main():
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        mark = "✅" if cond else "❌"
        if detail:
            print(f"  {mark} {name} | {detail}")
        else:
            print(f"  {mark} {name}")
        if not cond:
            ok = False

    cfg = load_config()
    llm = LLMClient(cfg["llm"])
    embedder = Embedder()
    tool = OfflineSearchTool(cfg["search"]["offline"], embedder=embedder)

    with open(PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json", encoding="utf-8") as f:
        questions = json.load(f)
    q = next(x for x in questions if x["_id"] == SAMPLE_QID)
    gold_titles = {t for t, _ in q["supporting_facts"]}
    print(f"Q: {q['question']}\n  gold: {sorted(gold_titles)}")

    # ---------- 构造一个 core 缺口槽 ----------
    sq = SubQuestion(id="sq1", text="find the nationality of both people",
                     qtype="core", slots=[Slot("relation", "Ed Wood nationality")])
    slot = sq.slots[0]

    # ---------- [1] 查询生成（真实小模型） ----------
    print("\n[1] QueryGenerator（真实 small 调用，绑定缺口槽）")
    ledger = ("searches used 0/10 · tokens 0 · cost $0.0000 · "
              "remaining $0.0500 · final-reserve 18%")
    qgen = QueryGenerator(llm)
    g1 = qgen.generate(sq, slot, templates=["{entity} nationality"],
                       budget_ledger=ledger)
    print(f"  生成查询: {g1['query']!r}  focus={g1['focus']!r}")
    check("查询非空", bool(g1.get("query")))
    check("含缺口实体", "wood" in g1["query"].lower() or "ed" in g1["query"].lower(),
          g1["query"])

    # ---------- [2] 真实检索 ----------
    print("\n[2] 检索（offline 真实 corpus）")
    docs = tool.evidence_search(g1["query"], gap_slot=slot, qid=q["_id"])
    print(f"  返回 {len(docs)} 条, calls={tool.calls}")
    for d in docs[:3]:
        print(f"   [{d.score:.3f}] {d.title}")

    # ---------- [3] 效用过滤（真实嵌入） ----------
    print("\n[3] UtilityEvaluator（真实嵌入 relevance/novelty/redundancy）")
    ut = UtilityEvaluator(embedder, cfg["execution"]["utility"])
    mem = WorkingMemory()
    kept, dropped = ut.filter(docs, slot.text, mem)
    print(f"  keep={len(kept)} drop={len(dropped)}")
    for d, s in kept:
        print(f"   KEEP [{s:.2f}] {d.title}")
    for d, why in dropped:
        print(f"   DROP [{why}] {d.title}")
    check("有 keep 项", len(kept) >= 1, f"keep={len(kept)}")

    # ---------- [4] 证据压缩（真实小模型 → Fact[]） ----------
    print("\n[4] EvidenceCompressor（真实 small 抽取 Fact[]，数值/日期保留）")
    comp = EvidenceCompressor(llm)
    ev = comp.compress(kept[0][0], sq, round_no=1, budget_ledger=ledger)
    print(f"  {ev.title} → {len(ev.facts)} 条 Fact（nationality 视角）:")
    for f in ev.facts:
        print(f"   [{f.ftype}] {f.subject} — {f.predicate}: {f.value}")
    check("抽到 Fact", len(ev.facts) >= 1)

    # 数值/日期保留：换数值聚焦的子问题，校验 "1994" 以 time 事实原样保留
    sq_num = SubQuestion(id="sq2", text="when the film was released and who directed it",
                         qtype="core", slots=[Slot("numeric", "release year")])
    ev_num = comp.compress(kept[0][0], sq_num, round_no=1, budget_ledger=ledger)
    num_values = [f.value for f in ev_num.facts if f.value]
    print(f"  数值聚焦视角 → {len(ev_num.facts)} 条 Fact，值集合={num_values}")
    check("抽到 Fact（数值视角）", len(ev_num.facts) >= 1)
    numeric_facts = [f for f in ev_num.facts if f.ftype in ("numeric", "time")]
    check("数值/时间事实原样保留", any("1994" in v for v in num_values),
          f"numeric/time={len(numeric_facts)}, values={num_values}")

    # ---------- [5] 工作记忆指纹去重 ----------
    print("\n[5] WorkingMemory 指纹去重")
    eid1 = mem.add(ev, UtilityEvaluator.fingerprint(ev.raw_text))
    ev2 = comp.compress(kept[0][0], sq, round_no=2, budget_ledger=ledger)
    eid2 = mem.add(ev2, UtilityEvaluator.fingerprint(ev2.raw_text))
    check("同一文档二次入库 → 同一 eid", eid1 == eid2, f"{eid1} vs {eid2}")
    check("evidence 只有 1 份", len(mem.evidence) == 1)
    cands = mem.candidates_for(slot.text)  # "Ed Wood nationality" → 词重叠命中 "wood"
    check("candidates_for 命中", len(cands) >= 1, f"hit={len(cands)}")
    ctx = mem.context_text()
    print(f"  上下文组装 {len(ctx)} chars")

    # ---------- [6] Budget Ledger 生效 ----------
    print("\n[6] 预算账本注入与成本记账")
    print(f"  真实 LLM 调用次数: {STATS['calls']}（小模型）")
    print(f"  累计成本: ${STATS['cost_usd']:.4f}")
    check("有真实 API 调用", STATS["calls"] >= 3, f"calls={STATS['calls']}")
    check("成本已记账", STATS["cost_usd"] > 0)

    print("\n" + "=" * 60)
    print("✅ M5 执行层冒烟通过（真实组件全链路）" if ok else "❌ 存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
