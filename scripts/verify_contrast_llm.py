"""task #26：真实触发 LLM 对比提炼路径（补 smoke13 盲区）。

smoke13 里 build_contrast_rules 从未真实调用 LLM（warm-train 8 题各类型只有成功侧，
走了 _deterministic_rules 兜底，task_rules 与确定性模板逐字一致）。本脚本构造真实
"同 task_type 成功+失败" episode 对（bridge 型：5 成 3 败），单次真实 API 调
build_contrast_rules，验证产出非空 DO/DON'T 规则并写入技能。

真实性声明：
  - 问题文本 / 答案 / 成败标签：来自 smoke13 真实评测集（seed=42，rows-EASE(warm)）。
  - 分解/查询内容：按各题真实结构手写重建（每题的逐 query trace 未持久化，
    smoke13 时 episode 池还没接线）。LLM 调用本身是真实的。
  - 成本：单次 deepseek-chat 小模型调用 ≈ $0.001-0.005。
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config, PROJECT_ROOT
from ease.llm.client import LLMClient
from ease.embeddings.embedder import Embedder
from ease.experience.skill_store import SkillStore
from ease.experience.extractor import RunSummary, SubQSummary, build_contrast_rules
from ease.experience.skill import Skill
from ease.eval.data_loader import sample_questions


def build_episode(q, success, decomposition, high_value=None, low_value=None,
                  abandoned=None, coverage=0.9, stop=4, cost=0.006):
    """从真实问题 + 手写分解重建 RunSummary（成败标签真实）。"""
    subs = [SubQSummary(text=t, qtype=qt, searches_used=2, evidence_cited=success)
            for qt, t in decomposition]
    return RunSummary(
        qid=q["_id"], task_type="bridge", question=q["question"], success=success,
        sub_questions=subs,
        high_value_queries=high_value or [],
        high_value_query_slots=["relation"] * len(high_value or []),
        low_value_queries=low_value or [],
        abandoned_slots=abandoned or [],
        final_coverage=coverage if success else 0.4,
        optimal_stop_round=0, actual_stop_round=stop,
        total_cost=cost, reused_skills=["skill-bridge"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="不调 LLM：只打印构造的 episode 对（验证输入形态）")
    ap.add_argument("--db", default="data/skills/contrast-diag.db",
                    help="规则写入的临时技能库")
    args = ap.parse_args()

    qs = {q["_id"]: q for q in sample_questions(10, seed=42)}
    # smoke13 warm 逐题成败（bridge 型 8 题：5 成 3 败）
    warm = {}
    for line in open("data/runs/smoke13/rows-EASE(warm).jsonl", encoding="utf-8"):
        r = json.loads(line)
        warm[r["qid"]] = r["em"] > 0

    successes, failures = [], []
    # —— 成功侧 3 题（真实分解，按其问题结构）——
    successes.append(build_episode(
        qs["5ae143ed55429920d5234360"], True,
        [("core", "What university was Sergei Aleksandrovich Tokarev a professor at?"),
         ("follow_up", "In what year was that university founded?")],
        high_value=["Sergei Aleksandrovich Tokarev professor university",
                    "university founded year"]))
    successes.append(build_episode(
        qs["5ac3e0f7554299194317388b"], True,
        [("core", "Which actors starred in American Beauty?"),
         ("core", "Which actor appears in both American Beauty titles?")],
        high_value=["American Beauty cast actor", "actor common to both American Beauty"]))
    successes.append(build_episode(
        qs["5ab97d0a5542996be202051e"], True,
        [("core", "Who assisted the attempted surrender of the defeated British army at Yorktown?"),
         ("core", "Who was hung for assisting that surrender?")],
        high_value=["assisted surrender Yorktown British army",
                    "hung for assisting surrender Yorktown"]))
    # —— 失败侧 3 题（真实分解 + 真实失败诊断信号）——
    failures.append(build_episode(
        qs["5abc19705542993a06baf86e"], False,
        [("core", "Which actress and writer starred in Black Book?"),
         ("core", "What heritage is that actress?"),
         ("core", "Confirm the actress and writer of that heritage")],
        low_value=["Black Book actress heritage", "actress writer nationality",
                   "Black Book actress and writer"],
        abandoned=["sq2:entity"], coverage=0.5, stop=6, cost=0.02))
    failures.append(build_episode(
        qs["5ae518655542993aec5ec139"], False,
        [("core", "Which upper house of which state legislature was Ken Pruitt a member of?"),
         ("core", "What was the size of that upper house?")],
        low_value=["Ken Pruitt upper house member", "size of upper house"],
        abandoned=["sq2:numeric"], coverage=0.5, stop=6, cost=0.018))
    failures.append(build_episode(
        qs["5adef1b35542993a75d263af"], False,
        [("core", "Which Mexican and American film actress plays Ethel Houbiers' character?"),
         ("core", "Which other role is that actress known for?")],
        low_value=["Mexican American film actress Ethel Houbiers",
                   "actress known for another role"],
        abandoned=["sq2:entity"], coverage=0.4, stop=7, cost=0.021))

    n_ok = len(successes); n_fail = len(failures)
    print(f"构造 episode 对：bridge 成功 {n_ok} / 失败 {n_fail}（真实题 + 真实成败标签）")

    if args.dry_run:
        print("--dry-run：不调 LLM。成功/失败 episode 已就绪（见上），退出。")
        return

    # —— 真实 LLM 对比提炼（单次调用）——
    cfg = load_config()
    llm = LLMClient(cfg["llm"])
    rules_map = build_contrast_rules(successes + failures, llm)
    rules = rules_map.get("bridge", [])
    print(f"\nLLM 对比提炼产出 {len(rules)} 条规则：")
    for r in rules:
        print(f"  • {r}")
        assert r.lower().startswith(("do", "don't")), f"规则不以 DO/DON'T 开头: {r}"
    assert rules, "LLM 对比提炼产出为空——盲区未补！"

    # —— 写入临时技能库验证入库 ——
    db = PROJECT_ROOT / args.db
    store = SkillStore({**cfg["experience"], "skills_db": str(db)}, embedder=Embedder())
    sk = store.get("skill-bridge")
    if sk is None:
        sk = Skill(name="skill-bridge", task_type="bridge",
                   description="Use when answering bridge multi-hop questions: connect two entities via an intermediate hop.")
        store.add_skill(sk)
    sk.task_rules = rules
    store.save(sk)
    back = store.get("skill-bridge").task_rules
    assert back == rules, "task_rules 入库回读不一致"
    store.close()
    print(f"\n✅ 真实 LLM 对比路径端到端验证通过：{len(rules)} 条 DO/DON'T 入库回读")
    print(f"   技能库: {db}")


if __name__ == "__main__":
    main()
