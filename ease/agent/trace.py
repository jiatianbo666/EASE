"""轨迹记录 —— demo 与评测共用（task.md §9.7）。
每个 run 一条 Trace，序列化后写入 run 目录 / 评测汇总。
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Trace:
    question: str = ""
    qid: str = ""
    task_feature: dict = field(default_factory=dict)     # 阶段1 特征
    prior_skill: str = ""                                # 命中技能名（无则空）
    sub_questions: list = field(default_factory=list)    # [{id,text,qtype}]
    events: list = field(default_factory=list)           # [{round,kind,detail}]
    searches: list = field(default_factory=list)         # [{query,focus,kept,dropped}]
    coverage_hist: list = field(default_factory=list)    # 每轮 total_coverage
    budget_used_searches: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    answer: str = ""
    cites: list = field(default_factory=list)
    unaddressed: list = field(default_factory=list)

    def event(self, round_no, kind, detail):
        self.events.append({"round": round_no, "kind": kind, "detail": detail})

    def search_record(self, query, focus, kept, dropped, stype="", anchors=None):
        self.searches.append({
            "query": query,
            "focus": focus,
            "kept": [d.title for d, _ in kept],
            "dropped": [(d.title, why) for d, why in dropped],
            "stype": stype,
            "anchors": list(anchors or []),   # 查询生成时使用的已解析实体（抽象化原料）
        })

    def to_dict(self):
        return asdict(self)
