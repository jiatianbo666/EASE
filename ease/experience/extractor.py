"""轨迹 → 技能提炼（P5 成功门控；task.md §9.8）。

输入：RunSummary（由 ease_agent 在每题结束后构建）。
产出：SkillUpdate 列表，由调用方应用到 SkillStore。
规则：
  - 失败 run：对复用过的技能 DOWNVOTE（reuse_failure）。
  - 成功 run：复用技能 UPVOTE；并从轨迹派生/更新一个任务技能
    （预算比例 ← 各子问题实际检索占比；分解模板 ← 子问题文本；
     高价值查询 ← 产出被引用证据的查询）。
"""
import re

from dataclasses import dataclass, field


@dataclass
class SubQSummary:
    text: str
    qtype: str              # core/background/follow_up
    searches_used: int
    evidence_cited: bool    # 该子问题证据是否被最终答案引用


@dataclass
class RunSummary:
    qid: str
    task_type: str
    question: str
    success: bool
    sub_questions: list = field(default_factory=list)   # list[SubQSummary]
    high_value_queries: list = field(default_factory=list)
    high_value_query_slots: list = field(default_factory=list)  # 平行：每条高价值查询的槽型 stype
    high_value_query_anchors: list = field(default_factory=list)  # 平行：每条查询用的已解析实体（抽象化原料）
    question_entities: list = field(default_factory=list)  # 该题 task_feature 的实体（抽象化原料）
    low_value_queries: list = field(default_factory=list)
    abandoned_slots: list = field(default_factory=list)  # ["sq2:entity", ...] 放弃的槽（成败诊断）
    final_coverage: float = 0.0
    optimal_stop_round: int = 0
    actual_stop_round: int = 0
    total_cost: float = 0.0
    reused_skills: list = field(default_factory=list)


def _abstract_query(query, entity_names):
    """把查询中的训练实体名抽象成 <entityN> 占位符（长串优先 + 按首次出现编号）。

    断点1修复：query_meta 存的是带训练实体名的具体查询（"Danny Boyle Slumdog ..."），
    注入给异题 = 喂噪音/泄漏。抽象成骨架后只保留槽型+结构，可跨题复用。
    仅抽象 ≥2 词的实体名（单 token 如 Dutch/1755 是答案/值词，抽象反而丢信息）。
    """
    if not query or not entity_names:
        return query
    q_low = query.lower()
    matched = []
    seen = set()
    for name in sorted(entity_names, key=lambda x: -len(x)):
        key = name.lower().strip()
        if key in seen or len(key.split()) < 2 or key not in q_low:
            continue
        seen.add(key)
        matched.append((q_low.index(key), name))
    matched.sort(key=lambda x: x[0])
    out = query
    for i, (_, name) in enumerate(matched, 1):
        out = re.sub(re.escape(name), f"<entity{i}>", out, flags=re.IGNORECASE)
    return out


@dataclass
class SkillUpdate:
    op: str                 # new | upvote | downvote | edit
    target: object          # Skill（new/edit）或 str 技能名（upvote/downvote）
    reason: str = ""


def derive_skill(run, base_name):
    """从成功 run 派生一个任务技能（task_type 相同者应合并到同一技能）。"""
    from .skill import Skill
    total_searches = sum(s.searches_used for s in run.sub_questions) or 1
    ratios = []
    for s in run.sub_questions:
        ratios.append(round(s.searches_used / total_searches, 3))
    # 归一化到和=1
    total = sum(ratios) or 1.0
    ratios = [round(r / total, 3) for r in ratios]

    # phase2：条目带 qtype；查询模板带来源槽型 stype（注入按 stype 裁剪的前提）
    decomp = [{"qtype": s.qtype, "text": s.text} for s in run.sub_questions]
    q_templates = run.high_value_queries[:5]
    # 断点1修复：抽象化原料 = 每条查询的已解析实体 + 该题任务实体（都可能是
    # 具体训练题实体名，注入异题会泄漏/污染）。
    anchors_lists = list(run.high_value_query_anchors) or []
    entity_pool = list(run.question_entities)
    for al in anchors_lists:
        entity_pool.extend(al)
    q_meta = []
    for i, q in enumerate(run.high_value_queries):
        st = run.high_value_query_slots[i] if i < len(run.high_value_query_slots) else ""
        if not st:
            continue
        q_meta.append({"stype": st, "text": _abstract_query(q, entity_pool)})
        if len(q_meta) >= 5:
            break

    # 从最优停止点推断 stop_params
    stop = {"optimal_stop_round": run.optimal_stop_round}
    desc = (f"Use when answering {run.task_type} multi-hop questions: "
            f"decompose into {len(decomp)} typed sub-questions, allocate "
            f"budget as {ratios}, and stop around round {run.optimal_stop_round}.")

    return Skill(
        name=base_name,
        description=desc,
        task_type=run.task_type,
        decomposition_template=decomp,
        budget_ratio=ratios,
        query_templates=q_templates,
        query_meta=q_meta,
        stop_params=stop,
        success_rate=1.0,
        avg_cost=run.total_cost,
        usage_count=1,
    )


def _failure_reason(run):
    """真实失败诊断（替代固定模板）：放弃的槽 + 覆盖卡死 + 零证据查询。"""
    parts = []
    if run.abandoned_slots:
        parts.append("槽放弃 " + ",".join(run.abandoned_slots))
    if run.final_coverage < 0.5:
        parts.append(f"覆盖卡在 {run.final_coverage:.2f}")
    if run.low_value_queries:
        parts.append("零证据查询 " + " / ".join(q[:40] for q in run.low_value_queries[:2]))
    detail = "；".join(parts) or "最终答案错误"
    return f"reuse_failure: {detail}"


def build_updates(run):
    """根据 run 摘要生成技能更新操作列表。"""
    updates = []
    if not run.success:
        reason = _failure_reason(run)   # phase2：真实诊断，喂给 failure_lessons
        for name in run.reused_skills:
            updates.append(SkillUpdate("downvote", name, reason=reason))
        return updates

    # 成功：复用技能点赞
    for name in run.reused_skills:
        updates.append(SkillUpdate("upvote", name, reason="reuse_success"))

    # 从成功轨迹派生任务技能
    base_name = f"skill-{run.task_type}"
    skill = derive_skill(run, base_name)
    if run.sub_questions:
        updates.append(SkillUpdate("new", skill, reason="verified_run"))
    return updates


def apply_updates(store, updates):
    """把 SkillUpdate 应用到 SkillStore，返回 (n_insert, n_merge, n_vote, n_del)。"""
    counts = {"new": 0, "merged": 0, "upvote": 0, "downvote": 0, "deleted": 0}
    for u in updates:
        if u.op == "new":
            res = store.add_skill(u.target, verified=True)
            counts["new" if res == "inserted" else "merged"] += 1
        elif u.op == "edit":
            res = store.add_skill(u.target, verified=True)
            counts["merged" if res == "merged" else "new"] += 1
        elif u.op == "upvote":
            if store.upvote(u.target):
                counts["upvote"] += 1
        elif u.op == "downvote":
            r = store.downvote(u.target, reason=u.reason)
            if r == "deleted":
                counts["deleted"] += 1
            else:
                counts["downvote"] += 1
    return counts


# ================= 成败对比提炼规则（phase2 Fix 3）=================

def _render_episode(run, kind):
    """把一次 run 摘要渲染成对比输入的轨迹片段（不含金标，防泄漏）。"""
    lines = [f"[{kind}] {run.question[:100]} (stop={run.actual_stop_round}, "
             f"cov={run.final_coverage:.2f}, ${run.total_cost:.4f})"]
    for s in run.sub_questions:
        lines.append(f"  - [{s.qtype}] {s.text[:90]} (searches={s.searches_used})")
    if run.high_value_queries:
        lines.append("  queries: " + " | ".join(q[:70] for q in run.high_value_queries[:4]))
    if kind == "FAIL":
        if run.abandoned_slots:
            lines.append("  abandoned: " + ", ".join(run.abandoned_slots))
        if run.low_value_queries:
            lines.append("  dead-end: " + " | ".join(q[:60] for q in run.low_value_queries[:3]))
    return "\n".join(lines)


def _llm_contrast(llm, task_type, successes, failures, budget_ledger=None):
    """一次小模型调用：成败轨迹对比 → DO/DON'T 规则。只收 DO/DON'T 开头的行。"""
    from ..llm.prompts import CONTRAST_RULES
    prompt = CONTRAST_RULES.format(
        task_type=task_type,
        n_success=len(successes),
        success_traces="\n\n".join(_render_episode(e, "OK") for e in successes),
        n_failure=len(failures),
        failure_traces="\n\n".join(_render_episode(e, "FAIL") for e in failures),
    )
    r = llm.chat_json("small", [{"role": "user", "content": prompt}],
                      schema_hint='{"rules":["string"]}', budget_ledger=budget_ledger)
    rules = []
    for x in (r or {}).get("rules") or []:
        t = str(x).strip()
        if len(t) > 120:
            t = t[:119] + "…"
        if re.match(r"^(do|don'?t)\b", t, re.IGNORECASE):
            rules.append(t)
    return rules[:6]


def _deterministic_rules(run):
    """仅成功侧时的兜底规则：从 trace 结构化信号生成，零 LLM。"""
    qtypes = [s.qtype for s in run.sub_questions]
    return [
        f"DO: 先拆 {len(qtypes)} 个子问题，顺序 [{', '.join(qtypes)}]，先解决 core 实体再整合。",
        "DO: 属性槽（数值/时间/关系）的检索查询必须携带已解析实体的完整名称。",
    ]


# ================= 断点3：episode 池持久化 =================

def episode_to_dict(run):
    """RunSummary → 可持久化 dict。不含金标/评测答案（_render_episode 同标准），防泄漏。"""
    return {
        "qid": run.qid,
        "task_type": run.task_type,
        "question": run.question,
        "success": bool(run.success),
        "sub_questions": [
            {"text": s.text, "qtype": s.qtype, "searches_used": s.searches_used,
             "evidence_cited": bool(s.evidence_cited)} for s in run.sub_questions],
        "high_value_queries": list(run.high_value_queries),
        "high_value_query_slots": list(run.high_value_query_slots),
        "high_value_query_anchors": [list(a) for a in run.high_value_query_anchors],
        "question_entities": list(run.question_entities),
        "low_value_queries": list(run.low_value_queries),
        "abandoned_slots": list(run.abandoned_slots),
        "final_coverage": run.final_coverage,
        "optimal_stop_round": run.optimal_stop_round,
        "actual_stop_round": run.actual_stop_round,
        "total_cost": run.total_cost,
        "reused_skills": list(run.reused_skills),
    }


def episode_from_dict(d):
    """episode_to_dict 的逆：dict → RunSummary（重建 build_contrast_rules 输入）。"""
    subs = [SubQSummary(text=s.get("text", ""), qtype=s.get("qtype", "core"),
                        searches_used=s.get("searches_used", 0),
                        evidence_cited=s.get("evidence_cited", False))
            for s in d.get("sub_questions", [])]
    return RunSummary(
        qid=d.get("qid", ""),
        task_type=d.get("task_type", "unknown"),
        question=d.get("question", ""),
        success=d.get("success", False),
        sub_questions=subs,
        high_value_queries=list(d.get("high_value_queries", [])),
        high_value_query_slots=list(d.get("high_value_query_slots", [])),
        high_value_query_anchors=[list(a) for a in d.get("high_value_query_anchors", [])],
        question_entities=list(d.get("question_entities", [])),
        low_value_queries=list(d.get("low_value_queries", [])),
        abandoned_slots=list(d.get("abandoned_slots", [])),
        final_coverage=d.get("final_coverage", 0.0),
        optimal_stop_round=d.get("optimal_stop_round", 0),
        actual_stop_round=d.get("actual_stop_round", 0),
        total_cost=d.get("total_cost", 0.0),
        reused_skills=list(d.get("reused_skills", [])),
    )


def build_contrast_rules(episodes, llm, budget_ledger=None):
    """成败对比提炼：按 task_type 分组。

    - 该类型同时有成功+失败 → LLM 对比提炼 DO/DON'T（成败差异最富信息）。
    - 该类型只有成功 → 确定性模板规则（零 LLM）。
    - 该类型只有失败 → 不产规则（无成功侧对照，避免把失败经验当正例）。
    返回 {task_type: [rule, ...]}。
    """
    groups = {}
    for e in episodes:
        groups.setdefault(e.task_type, []).append(e)
    out = {}
    for tt, eps in groups.items():
        succ = [e for e in eps if e.success]
        fail = [e for e in eps if not e.success]
        if succ and fail:
            rules = _llm_contrast(llm, tt, succ, fail, budget_ledger)
            out[tt] = rules or _deterministic_rules(succ[0])
        elif succ:
            out[tt] = _deterministic_rules(succ[0])
    return out
