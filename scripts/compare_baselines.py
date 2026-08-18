"""六方法同题对比表生成器（纯本地确定性计算，零 API）。

输入:
  data/runs/stable50/rows-{EASE(cold),EASE(warm)}.jsonl   (已满 50)
  data/runs/stable50-baselines/rows-{CoT,RAG-Once,IRCoT,ReAct}.jsonl
输出:
  - 汇总表（EM/F1/s/题/cost/题，按 qid join，只统计六方法都有行的题）
  - EASE(cold/warm) vs 各基线的配对胜败表（McNemar 风格）
  - 每题六方法答案矩阵（EM 标记）
  - 唯一赢家题清单

用法:
  python scripts/compare_baselines.py
"""
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.eval.metrics import em_f1

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EASE_DIR = os.path.join(BASE_DIR, "data", "runs", "stable50")
BL_DIR = os.path.join(BASE_DIR, "data", "runs", "stable50-baselines")

METHODS = ["EASE(cold)", "EASE(warm)", "CoT", "RAG-Once", "IRCoT", "ReAct"]
FILE_NAMES = {
    "EASE(cold)": "EASE(cold)",
    "EASE(warm)": "EASE(warm)",
    "CoT": "CoT",
    "RAG-Once": "RAG-Once",
    "IRCoT": "IRCoT",
    "ReAct": "ReAct",
}
PATHS = {m: os.path.join(EASE_DIR if m.startswith("EASE") else BL_DIR,
                         f"rows-{FILE_NAMES[m]}.jsonl") for m in METHODS}


def load(method):
    p = PATHS[method]
    if not os.path.exists(p):
        return None, []
    rows = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return p, rows


def main():
    data = {}
    for m in METHODS:
        p, rows = load(m)
        data[m] = rows
        tag = os.path.basename(p) if p else "文件缺失"
        print(f"{m:<12} {len(rows):>3}/50 rows  {tag}")

    # 求交集 qid：只对齐"已收满"的方法（len==max），部分方法排除但行数照报
    max_cnt = max((len(rows) for rows in data.values()), default=0)
    complete = [m for m in METHODS if len(data[m]) == max_cnt]
    qids = None
    for m in complete:
        s = {r["qid"] for r in data[m]}
        qids = s if qids is None else (qids & s)
    if qids is None:
        print("\n没有任何方法有行——退出")
        return
    done = [m for m in METHODS if data[m]]
    complete_set = set(complete)
    print(f"\n对齐 qids: {len(qids)} 题（{len(complete)} 个已收满方法，当前最大行数 {max_cnt}）"
          f"{'' if len(done) == len(METHODS) else ' —— 未完成方法: ' + ', '.join(set(METHODS)-set(done))}")

    by_q = {m: {r["qid"]: r for r in data[m]} for m in METHODS}

    # ---- 汇总表 ----
    print("\n" + "=" * 64)
    print("六方法汇总（对齐 %d 题）" % len(qids))
    print("=" * 64)
    print(f"{'method':<12} {'EM':>6} {'F1':>6} {'s/题':>6} {'$/题':>8} {'LLM/题':>7}")
    rows_out = []
    for m in METHODS:
        if m not in complete_set:
            print(f"{m:<12}  —— 未收满 ({len(data[m])}/50) ——")
            continue
        qs = [by_q[m][q] for q in sorted(qids)]
        em = sum(r["em"] for r in qs) / len(qs)
        f1 = sum(r["f1"] for r in qs) / len(qs)
        s_avg = sum(r["searches"] for r in qs) / len(qs)
        c_avg = sum(r["cost"] for r in qs) / len(qs)
        l_avg = sum(r["llm_calls"] for r in qs) / len(qs)
        print(f"{m:<12} {em:>6.3f} {f1:>6.3f} {s_avg:>6.2f} {c_avg:>8.4f} {l_avg:>7.1f}")
        rows_out.append((m, em, f1, s_avg, c_avg, l_avg))

    # ---- EASE vs 各基线配对 ----
    print("\n" + "=" * 64)
    print("配对胜败（EASE 赢 / 基线赢 / 都对 / 都错）—— 同题异答才算胜负")
    print("=" * 64)
    for base_m in ["EASE(warm)", "EASE(cold)"]:
        for other in [m for m in METHODS if m != base_m and m in complete_set]:
            w = l = both = neither = 0
            disc = 0
            for q in sorted(qids):
                a = by_q[base_m][q]["em"] > 0
                b = by_q[other][q]["em"] > 0
                if a and b: both += 1
                elif a and not b: w += 1
                elif not a and b: l += 1
                else: neither += 1
                if a != b: disc += 1
            print(f"  {base_m:<11} vs {other:<10} {w:>2}-{l:<2} (都对{both} 都错{neither})  discordant={disc}")

    # ---- 每题答案矩阵 ----
    print("\n" + "=" * 64)
    print("每题答案矩阵（✓=EM 命中，✗=未命中）")
    print("=" * 64)
    print(f"{'qid':<26} " + " ".join(f"{m[:7]:>8}" for m in METHODS))
    for q in sorted(qids):
        marks = []
        for m in METHODS:
            r = by_q[m].get(q)
            marks.append("✓" if r and r["em"] > 0 else "✗" if r else "·")
        print(f"{q:<26} " + " ".join(f"{x:>8}" for x in marks))

    # ---- 唯一赢家（该题只有此方法对）----
    print("\n" + "=" * 64)
    print("唯一赢家题（该题仅此方法 EM 命中）")
    print("=" * 64)
    for m in METHODS:
        if not data[m]:
            continue
        uniq = []
        for q in sorted(qids):
            hits = [x for x in METHODS if by_q[x].get(q, {}).get("em", 0) > 0]
            if len(hits) == 1 and hits[0] == m:
                r = by_q[m][q]
                uniq.append((q, r.get("question", "")[:70]))
        print(f"  {m:<12} 唯一赢 {len(uniq)} 题")
        for q, ques in uniq:
            print(f"      {q}  {ques}")

    # ---- 交叉一致性：同题跨方法答案一致率 ----
    print("\n" + "=" * 64)
    print("答案字符串一致率（同 qid 答案文本逐字相等）")
    print("=" * 64)
    base = "EASE(warm)"
    for other in [m for m in METHODS if m != base and m in complete_set]:
        agree = sum(1 for q in qids
                    if by_q[base][q]["answer"].strip().lower()
                    == by_q[other][q]["answer"].strip().lower())
        print(f"  {base} ~ {other:<10} 同答 {agree}/{len(qids)}")


if __name__ == "__main__":
    main()
