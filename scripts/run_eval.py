"""评测运行器 —— EASE(cold/warm) vs 4 基线，全真实运行。

用法:
  python scripts/run_eval.py --n 10             # 冒烟（默认 warm-train 4）
  python scripts/run_eval.py --n 100 --warm-train 20 --out data/runs/full
  # 正式（可断点续跑）

冷启动/热启动（task.md §10.1 防泄漏）：
  EASE(cold)：空技能库，跑评测集，不固化（纯"无经验"对照）
  EASE(warm) ：先用**另一批样本**（seed+1，排除评测集 qid）预热技能库，
               再跑评测集并固化（经验复用增益）
断点续跑：每题结果追加到 {out}/rows-{method}.jsonl，重跑跳过已完 qid。
"""
import argparse
import csv
import datetime
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config, PROJECT_ROOT
from ease.llm.client import LLMClient, reset_stats
from ease.embeddings.embedder import Embedder
from ease.search.offline import OfflineSearchTool
from ease.experience.skill_store import SkillStore
from ease.agent.ease_agent import EASEAgent
from ease.baselines.common import BaselineResult
from ease.baselines.methods import build_baselines
from ease.eval.data_loader import sample_questions
from ease.eval.metrics import em_f1
from ease.eval.harness import evaluate
from ease.eval.report import write_report

EVAL_DB_COLD = PROJECT_ROOT / "data" / "skills" / "skills-eval-cold.db"
EVAL_DB_WARM = PROJECT_ROOT / "data" / "skills" / "skills-eval-warm.db"


def make_ease_method(agent, store, solidify, episode_sink=None):
    """包装 EASE：run → （可选）按真实 EM/F1 门控固化 → 返回基线统一结果。

    episode_sink：断点3 用——把每次固化的 RunSummary 追加进列表，供 episode 池收集。
    """
    def run(q):
        result = agent.run(q["question"], qid=q["_id"])
        if solidify:
            em, f1 = em_f1(result.answer, q.get("answer", ""))
            success = (em > 0) or (f1 >= 1.0)
            _, run_summary = agent.solidify(result, success=success)   # 成功门控（P5）
            if episode_sink is not None and run_summary is not None:
                episode_sink.append(run_summary)
            if result.trace.prior_skill:
                store.record_usage(result.trace.prior_skill, success, result.trace.cost_usd)
        t = result.trace
        return BaselineResult(
            answer=result.answer,
            searches_used=t.budget_used_searches,
            llm_calls=t.llm_calls,
            cost_usd=t.cost_usd,
            events=t.events,
        )
    return run


def _fresh(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qids", default="",
                    help="逗号分隔的指定评测 qid（替代 seed 采样；--n 失效，规模=len(qids)）")
    ap.add_argument("--warm-train", type=int, default=8,
                    help="warm 预训练样本数（独立批次，seed+1；>=8 保证有失败轨迹供成败对比）")
    ap.add_argument("--out", default="data/runs/eval")
    ap.add_argument("--skip-ease", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--fresh-store", action="store_true",
                    help="删除现有评测技能库（冷启动从头）")
    args = ap.parse_args()

    cfg = load_config()
    reset_stats()
    out_dir = PROJECT_ROOT / args.out
    os.makedirs(out_dir, exist_ok=True)

    if args.qids:
        qid_list = [x.strip() for x in args.qids.split(",") if x.strip()]
        questions = sample_questions(0, seed=args.seed, qids=qid_list)
        print(f"评测集: 指定 {len(questions)} 道题 (qids={len(qid_list)})")
    else:
        questions = sample_questions(args.n, seed=args.seed)
        print(f"评测集: n={len(questions)} (seed={args.seed})")
    types = {}
    for q in questions:
        t = q.get("type") or q.get("_type") or "?"
        types[t] = types.get(t, 0) + 1
    print(f"  问题类型分布: {types}")

    if args.fresh_store:
        _fresh(EVAL_DB_COLD)
        _fresh(EVAL_DB_WARM)
        print("已删除评测技能库（冷启动）")

    embedder = Embedder()
    llm = LLMClient(cfg["llm"])
    search = OfflineSearchTool(cfg["search"]["offline"], embedder=embedder)
    methods = {}

    if not args.skip_ease:
        # 断点3：持久化 episode 池（跨轮累积经验，方案 A 周期增量提炼）
        from ease.experience.episode_pool import EpisodePool
        pool = EpisodePool(PROJECT_ROOT / "data" / "skills" / "episodes.jsonl")
        eval_episodes = []   # warm 评测 run 的 RunSummary 收集处

        # ---- warm 预训练（独立批次，防泄漏）----
        if args.warm_train > 0:
            eval_ids = {q["_id"] for q in questions}
            train_qs = sample_questions(args.warm_train, seed=args.seed + 1,
                                        exclude_qids=eval_ids)
            if not os.path.exists(EVAL_DB_WARM) or os.path.getsize(EVAL_DB_WARM) == 0:
                store_t = SkillStore({**cfg["experience"], "skills_db": str(EVAL_DB_WARM)},
                                     embedder=embedder)
                agent_t = EASEAgent(cfg, llm, search, store_t, embedder=embedder)
                print(f"\n[warm 预训练] 独立批次 {len(train_qs)} 题（seed={args.seed+1}）...")
                episodes = []
                for qi, q in enumerate(train_qs, 1):
                    res = agent_t.run(q["question"], qid=q["_id"])
                    em, f1 = em_f1(res.answer, q.get("answer", ""))
                    success = (em > 0) or (f1 >= 1.0)
                    _, run = agent_t.solidify(res, success=success)
                    episodes.append(run)
                    if res.trace.prior_skill:
                        store_t.record_usage(res.trace.prior_skill, success, res.trace.cost_usd)
                    print(f"  [{qi}/{len(train_qs)}] {q['_id']} em={int(em)} "
                          f"searches={res.trace.budget_used_searches} "
                          f"cost=${res.trace.cost_usd:.4f}")
                # 断点3：追加预热 episodes → 增量提炼对比规则（方案A，替代旧的只跑一次逻辑）
                pool.add(episodes)
                s = pool.refresh_contrast(store_t, llm)
                print(f"  预热技能库: {store_t.count()} 条技能 · 池 {s['episodes']} eps · "
                      f"提炼 {s['types']} 类型 / {s['rules']} 条规则 / {s['llm_calls']} LLM 调用")

        # ---- EASE(cold)：空库、不固化 ----
        _fresh(EVAL_DB_COLD)   # 冷启动恒为空
        store_c = SkillStore({**cfg["experience"], "skills_db": str(EVAL_DB_COLD)},
                             embedder=embedder)
        agent_c = EASEAgent(cfg, llm, search, store_c, embedder=embedder)
        methods["EASE(cold)"] = make_ease_method(agent_c, store_c, solidify=False)

        # ---- EASE(warm)：预热库、固化 ----
        store_w = SkillStore({**cfg["experience"], "skills_db": str(EVAL_DB_WARM)},
                             embedder=embedder)
        print(f"\n[EASE(warm)] 预热技能库 {store_w.count()} 条技能")
        agent_w = EASEAgent(cfg, llm, search, store_w, embedder=embedder)
        methods["EASE(warm)"] = make_ease_method(agent_w, store_w, solidify=True,
                                                 episode_sink=eval_episodes)

    if not args.skip_baselines:
        methods.update(build_baselines(llm, search, cfg["control"]))

    summary, all_rows = evaluate(methods, questions, out_dir=str(out_dir), label=f"n={args.n}")

    # 断点3：warm 评测结束后，把评测 episodes 追加进持久化池并增量提炼
    # （评测 run 只记录问题/分解/查询/停止轮，不含金标答案，无泄漏）。
    if not args.skip_ease and eval_episodes:
        added = pool.add(eval_episodes)
        s = pool.refresh_contrast(store_w, llm)
        print(f"\n[断点3] 评测追加 {added} eps → 池内 {s['episodes']} eps · "
              f"提炼 {s['types']} 类型 / {s['rules']} 条规则 / {s['llm_calls']} LLM 调用")

    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "n": args.n, "seed": args.seed, "warm_train": args.warm_train,
        "corpus": "HotpotQA dev distractor (offline)",
        "model_small": cfg["llm"]["small_model"],
        "model_large": cfg["llm"]["large_model"],
        "notes": ("全部真实 API 调用 + 真实检索，无 mock。"
                  "EM/F1 为 HotpotQA 官方口径；成本按官方分时价折算。"
                  "warm 技能库来自独立批次（seed+1，排除评测集 qid）。"),
    }
    write_report(summary, all_rows, meta, str(out_dir / "summary.md"))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "meta": meta}, f, ensure_ascii=False, indent=2)
    # 每题一行 CSV（report_table）
    with open(out_dir / "report_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "qid", "em", "f1", "searches", "llm_calls", "cost", "question"])
        for name, rows in all_rows.items():
            for r in rows:
                w.writerow([name, r["qid"], r["em"], r["f1"], r["searches"],
                            r["llm_calls"], r["cost"], r["question"]])

    print("\n" + "=" * 70)
    print("评测汇总:")
    for name, s in summary.items():
        print(f"  {name:<14} EM={s['em']:.3f} F1={s['f1']:.3f} "
              f"searches/题={s['searches_avg']:.2f} cost/题=${s['cost_avg_usd']:.4f}")
    print(f"\n报告: {out_dir / 'summary.md'}")
    print(f"表格: {out_dir / 'report_table.csv'}")


if __name__ == "__main__":
    main()
