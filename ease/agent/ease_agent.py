"""EASE 主循环（task.md §9 —— 四阶段：经验→规划→执行循环→答案+固化）。

全真实链路：
  1 经验检索    小模型任务特征 → SkillStore 向量检索 → prior skill
  2 预算规划    小模型分解子问题（技能分解 few-shot）→ BudgetPlanner 分配
  3 执行循环    Router(缺口绑定) → 查询生成 → 真实检索 → 效用过滤 →
                证据压缩 → 工作记忆去重 → 覆盖度判定 → 缺口再检测 → 停止检查
  4 答案+固化   大模型最终答案（带 [eid] 引用）→ 成功 run 技能固化（ExpeL 投票）

预算账本（BATS，P2）注入每一次 LLM 调用；停止带 safe-stop 监控（P8）。
"""
import re
from dataclasses import dataclass, field

from ..llm.client import STATS
from ..llm.prompts import (
    TASK_FEATURE, DECOMPOSE, SLOT_GAP, COVERAGE_JUDGE, CONFIDENCE,
    STOP_SAFE_MONITOR, VOI_ESTIMATE,
)
from ..control.metacognitive import MetacognitiveState, SubQuestion, Slot
from ..control.budget import BudgetPlanner
from ..control.router import Router
from ..control.stop import StopChecker
from ..execution.querygen import QueryGenerator
from ..execution.utility import UtilityEvaluator
from ..execution.compressor import EvidenceCompressor
from ..execution.memory import WorkingMemory
from ..experience.extractor import RunSummary, SubQSummary, build_updates, apply_updates
from .answer import generate_answer
from .trace import Trace

COVERAGE_HINT = '{"answered":bool,"supporting_evidence":["eid"]}'
SLOT_HINT = ('{"slots":[{"stype":"entity|relation|numeric|time","text":"string",'
             '"importance":0-1,"filled":bool}]}')
CONF_HINT = '{"proposed_answer":"string","p_true":0.0-1.0}'
SAFE_HINT = '{"any_core_unaddressed":bool,"any_conflict":bool}'
VOI_HINT = '{"p_change":0.0-1.0,"reason":"string"}'

# ---- 锚点感知（Fix A：hop-1 已解析实体注入依赖槽查询）----
_ENTITY_STOP = set(
    "the a an of and or to in on at for from by with as is are was were be been being "
    "this that these those it its he she they we you i me my your his her him them "
    "who what which where when how why not no".split()
)
_NUMBER_RE = re.compile(r"^\d[\d,.’']*\w*$")   # 纯数字/年份/带数量单位
_ANCHOR_SLOT_TYPES = ("numeric", "time", "relation")


def _looks_like_entity(s):
    """实体性启发：≥2 个有信息量的词、含大写、非纯数字/年份。
    单 token（如 Dutch / 1755）不进锚点；以 4 位年份结尾的日期值（如
    "January 25, 1755"）也不进 —— 避免日期值被当锚点注入检索查询。"""
    if not s:
        return False
    s = s.strip()
    if len(s) < 2 or _NUMBER_RE.match(s) or re.search(r"\d{4}$", s):
        return False
    toks = s.split()
    if len(toks) < 2 or not any(ch.isupper() for ch in s):
        return False
    sig = [t for t in toks if t.lower() not in _ENTITY_STOP]
    return len(sig) >= 2


def _anchor_ref_text(slot_text, slot_stype, anchors):
    """utility.filter 的相关性参考文本：属性槽未含锚点时追加 top anchor，
    否则 gold 文档可能因对通用槽文本低相关被 low_score 丢弃。"""
    if slot_stype not in _ANCHOR_SLOT_TYPES or not anchors:
        return slot_text
    low = slot_text.lower()
    if any(len(a.split()) >= 2 and a.lower() in low for a in anchors):
        return slot_text
    return f"{slot_text} {anchors[0]}"


@dataclass
class RunResult:
    answer: str = ""
    cites: list = field(default_factory=list)
    unaddressed: list = field(default_factory=list)
    trace: Trace = None
    state: MetacognitiveState = None
    memory: WorkingMemory = None
    ok: bool = False


class EASEAgent:
    def __init__(self, config, llm, search, skill_store, embedder=None):
        self.cfg = config
        self.llm = llm
        self.search = search
        self.store = skill_store
        from ..embeddings.embedder import Embedder
        self.embedder = embedder or Embedder()
        self.qgen = QueryGenerator(llm)
        self.utility = UtilityEvaluator(self.embedder, config["execution"]["utility"])
        self.compressor = EvidenceCompressor(llm)
        self.router = Router(config["control"])
        ctrl = config["control"]
        self.stop_checker = StopChecker(
            eps=ctrl.get("marginal_gain_eps", 0.02),
            window=ctrl.get("marginal_gain_window", 2),
            cov_stop=ctrl.get("coverage_stop_threshold", 0.8),
            conf_stop=ctrl.get("conf_stop_threshold", 0.75),
        )
        self.voi_threshold = ctrl.get("voi_threshold", 0.25)
        self.max_slot_attempts = ctrl.get("max_slot_attempts", 2)
        self._cost0 = 0.0
        self._calls0 = 0

    # ================= 四阶段主循环 =================
    def run(self, question, qid=None):
        self._cost0 = STATS["cost_usd"]
        self._calls0 = STATS["calls"]

        trace = Trace(question=question, qid=qid)

        # ---- 阶段1 经验检索 ----
        feature = self._task_feature(question, trace)
        prior = self._retrieve_skill(feature, trace)

        # ---- 阶段2 预算规划 ----
        sub_questions = self._decompose(question, prior, trace)
        budget = self._make_budget(sub_questions, prior)
        state = MetacognitiveState(question=question, sub_questions=sub_questions)
        memory = WorkingMemory()

        # ---- 阶段3 执行循环 ----
        self._loop(question, qid, state, budget, memory, prior, trace)

        # ---- 阶段4 最终答案 ----
        ans = generate_answer(self.llm, state, memory, question,
                              budget_ledger=self._ledger(budget))
        trace.answer = ans["answer"]
        trace.cites = ans["cites"]
        trace.unaddressed = ans["unaddressed"]
        trace.llm_calls = STATS["calls"] - self._calls0
        trace.cost_usd = STATS["cost_usd"] - self._cost0
        trace.budget_used_searches = budget.searches_used

        return RunResult(
            answer=ans["answer"], cites=ans["cites"], unaddressed=ans["unaddressed"],
            trace=trace, state=state, memory=memory, ok=bool(ans["answer"]),
        )

    # ---------------- 阶段1：经验 ----------------
    def _task_feature(self, question, trace):
        r = self.llm.chat_json(
            "small",
            [{"role": "user", "content": TASK_FEATURE.format(question=question)}],
            schema_hint=('{"task_type":"bridge|comparison|single|boolean|other",'
                         '"complexity":"low|medium|high","domain":"string",'
                         '"entities":["string"]}'),
            budget_ledger=self._ledger(),
        )
        feature = r or {"task_type": "other", "complexity": "medium",
                        "domain": "", "entities": []}
        trace.task_feature = {
            "task_type": str(feature.get("task_type", "other")),
            "complexity": str(feature.get("complexity", "medium")),
            "domain": str(feature.get("domain", "")),
            "entities": list(feature.get("entities") or []),
        }
        return trace.task_feature

    def _retrieve_skill(self, feature, trace):
        desc = (f"{feature['task_type']} multi-hop question in {feature['domain'] or 'general'}: "
                f"{', '.join(feature['entities'][:3]) or 'no entity'}")
        # phase2 门控：task_type 硬门（不同任务类型的技能结构上不可能注入）+ min_sim 兜底
        min_sim = self.cfg["experience"].get("min_sim", 0.15)
        skills = self.store.retrieve(desc, top_k=1, task_type=feature["task_type"], min_sim=min_sim)
        if skills:
            trace.prior_skill = skills[0].name
        return skills[0] if skills else None

    # ---------------- 阶段2：分解 + 预算 ----------------
    def _decompose(self, question, prior, trace):
        few_shot = ""
        if prior:
            parts = []
            if prior.decomposition_template:
                qtypes = [s.get("qtype", "core") if isinstance(s, dict) else "core"
                          for s in prior.decomposition_template]
                parts.append(f"历史结构：先拆 {len(qtypes)} 个子问题，顺序 [{', '.join(qtypes)}]")
            if prior.task_rules:
                parts.append("历史成败规则：\n" + "\n".join(f"- {r}" for r in prior.task_rules[:5]))
            if parts:
                few_shot = ("参考历史经验（同任务类型，只作结构约束，勿照抄子问题原文）：\n"
                            + "\n".join(parts))
        r = self.llm.chat_json(
            "small",
            [{"role": "user", "content": DECOMPOSE.format(few_shot=few_shot, question=question)}],
            schema_hint='{"sub_questions":[{"text":"string","qtype":"core|background|follow_up"}]}',
            budget_ledger=self._ledger(),
        )
        subs = []
        if r and isinstance(r.get("sub_questions"), list):
            for i, s in enumerate(r["sub_questions"]):
                if not isinstance(s, dict) or not s.get("text"):
                    continue
                qtype = str(s.get("qtype", "core"))
                if qtype not in ("core", "background", "follow_up"):
                    qtype = "core"
                subs.append({"id": f"sq{i + 1}", "text": str(s["text"]), "qtype": qtype})
        if not subs:
            # 兜底：整题作为单个 core 子问题（真实失败时才走，不伪造）
            subs = [{"id": "sq1", "text": question, "qtype": "core"}]
        trace.sub_questions = subs
        return [SubQuestion(id=s["id"], text=s["text"], qtype=s["qtype"]) for s in subs]

    def _make_budget(self, sub_questions, prior):
        ctrl = self.cfg["control"]
        budget = BudgetPlanner(
            total_searches=ctrl.get("search_budget", 10),
            total_tokens=ctrl.get("search_token_budget", 200_000),
            finalization_reserve=self.cfg["llm"].get("finalization_reserve", 0.18),
            conflict_allowance=ctrl.get("conflict_allowance", 2),
        )
        budget.plan(sub_questions, skill_prior=prior)
        return budget

    # ---------------- 阶段3：执行循环 ----------------
    def _resolved_entities(self, memory, question):
        """从工作记忆事实确定性提取已解析实体（hop-1 的答案实体），best first。

        排序 key：① 不在原问题中（= 检索才解析出来的实体，优先于问题自身实体）
        ② ftype∈(entity,relation) ③ 证据新近度 ④ 出现频次。纯 python，无 LLM。
        """
        if not memory or not memory.evidence:
            return []
        qlow = question.lower()
        counts = {}            # 归一化实体串 -> 出现频次
        meta = {}              # 实体串 -> (not_in_question, ftype_prio, added_round)
        for ev in memory.evidence:
            for f in ev.facts:
                for cand in (f.subject, f.value):
                    if not _looks_like_entity(cand):
                        continue
                    key = re.sub(r"\s+", " ", cand).strip()
                    if not key:
                        continue
                    counts[key] = counts.get(key, 0) + 1
                    not_in_q = key.lower() not in qlow
                    ftype_prio = 1 if f.ftype in ("entity", "relation") else 0
                    prev = meta.get(key)
                    if (prev is None or not_in_q > prev[0]
                            or (not_in_q == prev[0] and ftype_prio > prev[1])):
                        meta[key] = (not_in_q, ftype_prio, ev.added_round)
                    elif not_in_q == prev[0] and ftype_prio == prev[1]:
                        meta[key] = (not_in_q, ftype_prio,
                                     max(prev[2], ev.added_round))
        ordered = sorted(meta.items(),
                         key=lambda kv: (kv[1][0], kv[1][1], kv[1][2], counts[kv[0]]),
                         reverse=True)
        return [k for k, _ in ordered]

    def _loop(self, question, qid, state, budget, memory, prior, trace):
        slot_attempts = {}          # key=(sq.id, stype) → 无覆盖度进展的连续检索次数
        abandoned = {}              # sq.id → set(stype)：已放弃的槽类型（跨轮次持久）
        raw_fallback_used = False   # 整题仅一次：槽放弃时用原始问题兜底检索
        round_no = 0
        ledger = self._ledger(budget)
        # 首轮：无证据 → 先做一次缺口检测（确定初始检索目标）
        self._refresh_slots(state, memory, ledger, trace, abandoned)

        while True:
            round_no += 1
            ledger = self._ledger(budget)
            gaps = state.detect_gaps()
            action = self.router.decide(state, budget, gaps=gaps)
            trace.event(round_no, "router", f"{action.kind}: {action.reason[:90]}")
            if action.kind != "RETRIEVE":
                trace.event(round_no, "stop", f"router → {action.kind}: {action.reason[:80]}")
                break

            sq, slot = action.gap
            key = (sq.id, slot.stype)

            # VOI 门控（P10）：非 core 缺口且预算紧张 → 先估信息价值
            if sq.qtype != "core" and budget.remaining_searches() <= 2:
                voi = self._voi(slot, state, ledger)
                if voi is not None and voi < self.voi_threshold:
                    trace.event(round_no, "voi_skip", f"p_change={voi:.2f} < {self.voi_threshold}")
                    break

            # 锚点感知（Fix A）：从记忆提取已解析实体，注入查询 + 相关性参考文本
            anchors = self._resolved_entities(memory, question)
            ref_text = _anchor_ref_text(slot.text, slot.stype, anchors)

            # 同一槽类型连续无覆盖度进展 → 放弃（防死循环；slot 文本每轮漂移，
            # 必须按 stype 计数 + 按覆盖度判定进展，否则永不触发）。
            # Fix A：放弃前整题**一次**用原始问题兜底检索（RAG-Once 实证的有效查询），
            # 走与正常轮相同的检索→过滤→压缩→覆盖度管道，给答案生成留证据。
            if slot_attempts.get(key, 0) >= self.max_slot_attempts:
                slot.filled = True
                abandoned.setdefault(sq.id, set()).add(slot.stype)
                if (not raw_fallback_used
                        and not self.qgen.has_seen(self.qgen.fingerprint(question))):
                    raw_fallback_used = True
                    q = {"query": question, "focus": "raw-question fallback"}
                    trace.event(round_no, "slot_abandoned",
                                f"{sq.id}:{slot.stype} attempts={slot_attempts[key]} 无进展 → 原始问题兜底")
                else:
                    trace.event(round_no, "slot_abandoned",
                                f"{sq.id}:{slot.stype} attempts={slot_attempts[key]} 无进展")
                    continue
            else:
                # phase2：查询模板只注入与当前槽型匹配的（query_meta 无匹配 → 不注入，
                # 纯槽驱动，宁缺勿滥——防别的槽型的模板污染本槽查询）
                q_templates = None
                if prior and prior.query_meta:
                    q_templates = [m["text"] for m in prior.query_meta
                                   if m.get("stype") == slot.stype][:3]
                q = self.qgen.generate(sq, slot, templates=q_templates,
                                       resolved_entities=anchors,
                                       budget_ledger=ledger)
                if not q:
                    slot_attempts[key] = slot_attempts.get(key, 0) + 1
                    continue

            docs = self.search.evidence_search(q["query"], gap_slot=slot, qid=qid)
            budget.consume_search()
            sq.searches_used += 1

            kept, dropped = self.utility.filter(docs, ref_text, memory)
            trace.search_record(q["query"], q["focus"], kept, dropped,
                                stype=slot.stype, anchors=anchors)

            new_count = 0
            for doc, score in kept:
                ev = self.compressor.compress(doc, sq, round_no, budget_ledger=ledger)
                fp = UtilityEvaluator.fingerprint(ev.raw_text)
                eid = memory.add(ev, fp)
                if eid == ev.eid:      # 新入库
                    new_count += 1

            # 覆盖度判定（真实小模型）—— 用其判断本轮是否有实质进展
            judge = self._coverage_judge(ledger)
            cov_before = sq.coverage
            state.update_coverage(memory.evidence, coverage_judge=judge)
            cov_after = sq.coverage
            if cov_after > cov_before:
                slot_attempts[key] = 0      # 该槽类型取得实质进展 → 重置计数
                self._refresh_slots(state, memory, ledger, trace, abandoned)
            else:
                slot_attempts[key] = slot_attempts.get(key, 0) + 1   # 无进展 → 计数（无论是否有新证据入库）
            trace.coverage_hist.append(round(state.total_coverage(), 3))

            # 停止检查（复合 + safe-stop）
            conf = self._confidence(state, memory, ledger)
            self.stop_checker.observe_coverage(state.total_coverage())
            stop, reason = self.stop_checker.should_stop(
                state, budget, safe_monitor=self._safe_monitor(memory, ledger),
                calib_confidence=conf)
            trace.event(round_no, "stop_check",
                        f"stop={stop} cov={state.total_coverage():.2f} conf={conf:.2f} | {reason[:90]}")
            if stop or budget.budget_exhausted():
                trace.event(round_no, "stop", reason or "search budget exhausted (wrap-up)")
                break

    # ---------------- LLM 判定回调 ----------------
    def _refresh_slots(self, state, memory, ledger, trace, abandoned=None):
        """为 open 子问题做真实 SLOT_GAP，更新 sq.slots。
        abandoned: {sq.id: set(stype)} —— 已放弃的槽类型强制 filled（跨轮次持久）。"""
        abandoned = abandoned or {}
        for sq in state.open_sub_questions():
            ev = "\n".join(e.compact(max_chars=400) for e in memory.evidence[:6]) or "(无证据)"
            r = self.llm.chat_json(
                "small",
                [{"role": "user", "content": SLOT_GAP.format(subq=sq.text, evidence=ev)}],
                schema_hint=SLOT_HINT, budget_ledger=ledger,
            )
            if not r or not isinstance(r.get("slots"), list):
                continue
            slots = []
            for s in r["slots"]:
                if not isinstance(s, dict) or not s.get("text"):
                    continue
                stype = str(s.get("stype", "entity"))
                slots.append(Slot(
                    stype=stype,
                    text=str(s["text"]),
                    importance=float(s.get("importance", 1.0)),
                    filled=bool(s.get("filled", False)) or stype in abandoned.get(sq.id, set()),
                ))
            if slots:
                sq.slots = slots

    def _coverage_judge(self, ledger):
        """覆盖度裁判：要求 answered=true 时必须给出记忆中真实存在的证据 eid
        （防小模型凭空判覆盖 —— Q2 教训：一篇无 heritage 的电影文被判覆盖）。"""
        def judge(subq_text, ev_texts):
            r = self.llm.chat_json(
                "small",
                [{"role": "user",
                  "content": COVERAGE_JUDGE.format(subq=subq_text,
                                                   evidence_blocks="\n".join(ev_texts) or "(无证据)")}],
                schema_hint=COVERAGE_HINT, budget_ledger=ledger,
            )
            r = r or {"answered": False, "supporting_evidence": []}
            known = set()
            for t in ev_texts:
                m = re.match(r"\[(ev_\d+)\]", t)
                if m:
                    known.add(m.group(1))
            supp = [e for e in (r.get("supporting_evidence") or []) if e in known]
            if r.get("answered") and not supp:
                return {"answered": False, "supporting_evidence": []}
            return {"answered": bool(r.get("answered")), "supporting_evidence": supp}
        return judge

    def _confidence(self, state, memory, ledger):
        """对 open core 子问题做真实置信探测，写回 sq.confidence；无 open core 时用已探测均值。"""
        probes = []
        for sq in state.open_sub_questions():
            if sq.qtype != "core":
                continue
            ev_ids = set(sq.evidence_ids)
            ev = "\n".join(e.compact(max_chars=400) for e in memory.evidence
                           if e.eid in ev_ids or not ev_ids)
            r = self.llm.chat_json(
                "small",
                [{"role": "user", "content": CONFIDENCE.format(subq=sq.text, evidence=ev or "(无证据)")}],
                schema_hint=CONF_HINT, budget_ledger=ledger,
            )
            if r:
                try:
                    p = float(r.get("p_true", 0.0))
                except (TypeError, ValueError):
                    p = 0.0
                p = max(0.0, min(1.0, p))
                sq.confidence = p
                probes.append(p)
        if not probes:
            known = [s.confidence for s in state.sub_questions if s.confidence > 0]
            return (sum(known) / len(known)) if known else 0.0
        return (sum(probes) / len(probes))

    def _safe_monitor(self, memory, ledger):
        def monitor(state):
            ev = "\n".join(e.compact(max_chars=300) for e in memory.evidence[:5]) or "(无证据)"
            r = self.llm.chat_json(
                "small",
                [{"role": "user", "content": STOP_SAFE_MONITOR.format(evidence_summary=ev)}],
                schema_hint=SAFE_HINT, budget_ledger=ledger,
            )
            return r or {"any_core_unaddressed": True, "any_conflict": False}
        return monitor

    def _voi(self, slot, state, ledger):
        r = self.llm.chat_json(
            "small",
            [{"role": "user", "content": VOI_ESTIMATE.format(
                slot_text=slot.text,
                state_summary=f"core_coverage={state.core_coverage():.2f}")}],
            schema_hint=VOI_HINT, budget_ledger=ledger,
        )
        if not r:
            return None
        try:
            return float(r.get("p_change", 0.0))
        except (TypeError, ValueError):
            return None

    # ---------------- 经验固化 ----------------
    def solidify(self, result, success):
        """按成功与否把 run 固化为技能更新（P5；成功门控由评测给定）。"""
        trace = result.trace
        sq_summaries = []
        for sq in result.state.sub_questions:
            cited = any(eid in trace.cites for eid in sq.evidence_ids)
            sq_summaries.append(SubQSummary(
                text=sq.text, qtype=sq.qtype, searches_used=sq.searches_used,
                evidence_cited=cited))
        kept = [s for s in trace.searches if s["kept"]]
        abandoned = [ev["detail"].split(" attempts=")[0]
                     for ev in trace.events if ev["kind"] == "slot_abandoned"]
        run = RunSummary(
            qid=trace.qid or trace.question,
            task_type=trace.task_feature.get("task_type", "unknown"),
            question=trace.question,
            success=success,
            sub_questions=sq_summaries,
            high_value_queries=[s["query"] for s in kept],
            high_value_query_slots=[s.get("stype", "") for s in kept],
            high_value_query_anchors=[s.get("anchors", []) for s in kept],
            question_entities=list(trace.task_feature.get("entities", [])),
            low_value_queries=[s["query"] for s in trace.searches if not s["kept"]],
            abandoned_slots=abandoned,
            final_coverage=trace.coverage_hist[-1] if trace.coverage_hist else 0.0,
            optimal_stop_round=0,
            actual_stop_round=len(trace.events),
            total_cost=trace.cost_usd,
            reused_skills=[trace.prior_skill] if trace.prior_skill else [],
        )
        updates = build_updates(run)
        counts = apply_updates(self.store, updates)
        return counts, run

    # ---------------- 内部 ----------------
    def _ledger(self, budget=None):
        if budget is None:
            return "searches used 0/10 · tokens 0 · cost $0.0000 · remaining 10"
        cost = STATS["cost_usd"] - self._cost0
        return budget.ledger() + f" · cost ${cost:.4f}"
