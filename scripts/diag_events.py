"""把指定 qid 前缀在 smoke10b/smoke10c 各运行的答案 + events dump 出来。

用法: python scripts/diag_events.py <qid前缀>   (如 5ae143ed)
events 里是 router/stop_check/slot_abandoned/stop 等元认知决策，配合
diag_retrieval.py 的 evidence 层输出，可定位失败在哪一步。
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")

P = sys.argv[1] if len(sys.argv) > 1 else None
RUNS = ["smoke10b", "smoke10c"]
METHODS = ["EASE(cold)", "EASE(warm)", "RAG-Once"]

for run in RUNS:
    for m in METHODS:
        p = f"data/runs/{run}/rows-{m}.jsonl"
        try:
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                if P and not r["qid"].startswith(P):
                    continue
                print(f"==== {run} / {m} ====")
                print(f"  question: {r['question'][:120]}")
                print(f"  gold={r['gold']!r}  answer={r['answer']!r}  em={r['em']}  "
                      f"f1={r['f1']:.2f}  searches={r['searches']}  "
                      f"llm_calls={r['llm_calls']}  cost=${r['cost']:.4f}")
                for ev in r.get("events", []):
                    print(f"  r{ev.get('round')} {ev.get('kind')} | {str(ev.get('detail'))[:170]}")
                print()
        except FileNotFoundError:
            pass
