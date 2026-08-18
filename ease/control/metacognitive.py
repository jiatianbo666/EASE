"""元认知状态 —— 覆盖度图谱 / 缺口列表 / 置信向量 / 冲突状态（task.md §6.1 §7.6）。

纯数据 + 状态机。LLM 判定（覆盖度、缺口检测）通过注入的 judge 回调实现，
agent 层把 judge 接到真实小模型（prompts.py 的 COVERAGE_JUDGE / SLOT_GAP）。
"""
from dataclasses import dataclass, field

CORE_W = 1.0
BACKGROUND_W = 0.5
FOLLOWUP_W = 0.3
WTYPE = {"core": CORE_W, "background": BACKGROUND_W, "follow_up": FOLLOWUP_W}


@dataclass
class Slot:
    stype: str                # entity | relation | numeric | time
    text: str                 # 需要的信息描述
    importance: float = 1.0
    filled: bool = False
    supporting_evidence: list = field(default_factory=list)


@dataclass
class SubQuestion:
    id: str
    text: str
    qtype: str = "core"
    weight: float = CORE_W
    coverage: float = 0.0
    confidence: float = 0.0
    status: str = "open"      # open | covered | conflict | abandoned
    evidence_ids: list = field(default_factory=list)
    slots: list = field(default_factory=list)
    searches_used: int = 0

    def __post_init__(self):
        self.weight = WTYPE.get(self.qtype, CORE_W)


@dataclass
class ConflictRecord:
    sub_q_id: str
    sides: list = field(default_factory=list)   # [[evidence_ids], ...]
    verdict: str = "undecided"


@dataclass
class MetacognitiveState:
    question: str = ""
    sub_questions: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    answer_draft: str = None
    status: str = "running"    # running | wrapping_up | done

    # ---------- 覆盖度 ----------
    def core_coverage(self):
        core = [s for s in self.sub_questions if s.qtype == "core"]
        if not core:
            return 0.0
        return sum(s.weight * s.coverage for s in core) / sum(s.weight for s in core)

    def total_coverage(self):
        if not self.sub_questions:
            return 0.0
        return sum(s.weight * s.coverage for s in self.sub_questions) / sum(s.weight for s in self.sub_questions)

    def open_sub_questions(self):
        return [s for s in self.sub_questions if s.status not in ("covered", "abandoned")]

    # ---------- 覆盖度更新（P8：必须有证据块支撑） ----------
    def update_coverage(self, evidence, coverage_judge=None):
        """判定每个 open 子问题是否被≥1证据块支撑。
        coverage_judge(subq_text, evidence_texts) -> {"answered":bool,"supporting_evidence":[eid]}
        judge=None 时用启发式兜底（仅测试/降级）：有证据即算覆盖。
        """
        for sq in self.open_sub_questions():
            if coverage_judge is not None:
                ev_texts = [f"[{e.eid}] {e.raw_text[:400]}" for e in evidence]
                verdict = coverage_judge(sq.text, ev_texts)
                answered = bool(verdict.get("answered"))
                supp = verdict.get("supporting_evidence") or []
            else:
                answered = bool(evidence)
                supp = [e.eid for e in evidence[:1]]
            if answered:
                sq.coverage = 1.0
                sq.evidence_ids = supp
                sq.status = "covered"
            else:
                sq.coverage = 0.0
                sq.status = "open"
        return self

    # ---------- 冲突 ----------
    def add_conflict(self, sub_q_id, side_evidence_ids):
        rec = ConflictRecord(sub_q_id=sub_q_id, sides=side_evidence_ids)
        if not any(c.sub_q_id == sub_q_id for c in self.conflicts):
            self.conflicts.append(rec)
        for sq in self.sub_questions:
            if sq.id == sub_q_id:
                sq.status = "conflict"
        return rec

    def has_unresolved_core_conflict(self):
        core_ids = {s.id for s in self.sub_questions if s.qtype == "core"}
        return any(c.sub_q_id in core_ids and c.verdict == "undecided"
                   for c in self.conflicts)

    # ---------- 缺口检测（slot-filling） ----------
    def detect_gaps(self, gap_detector=None):
        """返回 [(subq, slot), ...] 按 权重×重要性 降序。
        gap_detector(sq) -> [Slot,...]；不传时用子问题自身 slots。
        """
        gaps = []
        for sq in self.open_sub_questions():
            slots = gap_detector(sq) if gap_detector else sq.slots
            for sl in slots:
                if not sl.filled:
                    gaps.append((sq, sl))
        gaps.sort(key=lambda g: -(g[0].weight * g[1].importance))
        return gaps

    def core_gaps(self, gaps):
        return [g for g in gaps if g[0].qtype == "core"]
