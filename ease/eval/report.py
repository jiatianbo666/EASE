"""评测报告 —— 质量×效率双轴汇总（task.md §11）。纯确定性渲染。"""
import datetime


def write_report(summary, all_rows, meta, out_path):
    """meta: {n, seed, corpus, model_small, model_large, date, notes}"""
    lines = []
    lines.append(f"# EASE 评测报告 — 质量×效率双轴")
    lines.append("")
    lines.append(f"- 日期: {meta['date']}")
    lines.append(f"- 样本: n={meta['n']} · seed={meta['seed']} · 语料: {meta['corpus']}")
    lines.append(f"- 模型: small={meta['model_small']} · large={meta['model_large']}")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| 方法 | EM | F1 | 检索次数/题 | LLM调用/题 | 成本/题($) | 总成本($) |")
    lines.append("|------|----|----|------|--------|--------|--------|")
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['em']:.3f} | {s['f1']:.3f} | {s['searches_avg']:.2f} | "
            f"{s['llm_calls_avg']:.1f} | {s['cost_avg_usd']:.4f} | {s['total_cost_usd']:.4f} |")
    lines.append("")
    lines.append("## 双轴对比")
    lines.append("")
    lines.append("| 方法 | 质量(EM) | 效率(检索/题) | 成本(美元/题) | 每 EM 成本 |")
    lines.append("|------|------|----------|----------|----------|")
    for name, s in summary.items():
        per_em = (s["cost_avg_usd"] / s["em"]) if s["em"] > 0 else float("inf")
        per_em_s = f"{per_em:.4f}" if per_em != float("inf") else "∞(0 EM)"
        lines.append(
            f"| {name} | {s['em']:.3f} | {s['searches_avg']:.2f} | {s['cost_avg_usd']:.4f} | "
            f"{per_em_s} |")
    lines.append("")

    # 逐方法示例
    lines.append("## 逐题样例")
    lines.append("")
    for name, rows in all_rows.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| qid | EM | F1 | 检索 | 成本 | 问题 | 答案 |")
        lines.append("|-----|----|----|------|------|------|------|")
        for r in rows[:5]:
            ans = (r["answer"] or "")[:40].replace("|", "/")
            q = (r["question"] or "")[:40].replace("|", "/")
            lines.append(f"| {r['qid'][:10]} | {r['em']:.0f} | {r['f1']:.2f} | "
                         f"{r['searches']} | ${r['cost']:.4f} | {q} | {ans} |")
        lines.append("")
    if meta.get("notes"):
        lines.append("## 说明")
        lines.append("")
        lines.append(meta["notes"])
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
