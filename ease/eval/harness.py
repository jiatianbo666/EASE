"""评测 harness —— 逐题运行方法、算指标、增量落盘、可断点续跑。

真实运行（无 mock）；每题结果写入 {out_dir}/{name}.jsonl，
重跑时自动跳过已完成的 qid（正式 n=100 跑量需要）。
"""
import json
import os
import statistics
import time

from .metrics import em_f1


def _row_path(out_dir, name):
    return os.path.join(out_dir, f"rows-{name}.jsonl")


def load_done_qids(out_dir, name):
    path = _row_path(out_dir, name)
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["qid"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_row(out_dir, name, row):
    os.makedirs(out_dir, exist_ok=True)
    with open(_row_path(out_dir, name), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(methods, questions, out_dir=None, label=""):
    """methods: {name: fn(question) -> BaselineResult}。
    返回 (summary, all_rows)。summary 为 {name: {em,f1,searches,llm_calls,cost,...}}。
    """
    all_rows = {}
    for name, fn in methods.items():
        done = load_done_qids(out_dir, name) if out_dir else set()
        print(f"\n=== {name} [{label}] (已完 {len(done)}/{len(questions)}) ===")
        rows = []
        if out_dir and done:
            with open(_row_path(out_dir, name), encoding="utf-8") as f:
                rows = [json.loads(l) for l in f]
        for i, q in enumerate(questions, 1):
            qid = q["_id"]
            if qid in done:
                continue
            try:
                t0 = time.time()
                res = fn(q)
                em, f1 = em_f1(res.answer, q.get("answer", ""))
                row = {"qid": qid, "question": q["question"], "answer": res.answer,
                       "gold": q.get("answer", ""), "em": em, "f1": round(f1, 4),
                       "searches": res.searches_used, "llm_calls": res.llm_calls,
                       "cost": round(res.cost_usd, 6), "events": res.events}
                rows.append(row)
                if out_dir:
                    append_row(out_dir, name, row)
                print(f"  [{i}/{len(questions)}] {qid} em={int(em)} f1={f1:.3f} "
                      f"searches={res.searches_used} calls={res.llm_calls} "
                      f"cost=${res.cost_usd:.4f} ({time.time()-t0:.1f}s)")
            except Exception as e:
                row = {"qid": qid, "question": q["question"], "answer": "",
                       "gold": q.get("answer", ""), "em": 0.0, "f1": 0.0,
                       "searches": 0, "llm_calls": 0, "cost": 0.0,
                       "events": [{"kind": "error", "detail": str(e)}]}
                rows.append(row)
                if out_dir:
                    append_row(out_dir, name, row)
                print(f"  [{i}/{len(questions)}] {qid} ERROR: {e}")
        all_rows[name] = rows
    return aggregate(all_rows), all_rows


def _mean(xs):
    return round(statistics.mean(xs), 4) if xs else 0.0


def _stdev(xs):
    return round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0


def aggregate(all_rows):
    summary = {}
    for name, rows in all_rows.items():
        ems = [r["em"] for r in rows]
        f1s = [r["f1"] for r in rows]
        ss = [r["searches"] for r in rows]
        cs = [r["llm_calls"] for r in rows]
        costs = [r["cost"] for r in rows]
        summary[name] = {
            "n": len(rows),
            "em": _mean(ems),
            "em_std": _stdev(ems),
            "f1": _mean(f1s),
            "f1_std": _stdev(f1s),
            "searches_avg": _mean(ss),
            "searches_std": _stdev(ss),
            "llm_calls_avg": _mean(cs),
            "cost_avg_usd": _mean(costs),
            "total_cost_usd": round(sum(costs), 4),
            "em_hit": sum(ems),
        }
    return summary
