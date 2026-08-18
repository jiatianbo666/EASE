"""基线公共组件：结果结构 + 成本归因 + 真实检索计数。"""
from dataclasses import dataclass, field

from ..llm.client import STATS


@dataclass
class BaselineResult:
    answer: str = ""
    searches_used: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    events: list = field(default_factory=list)   # 供 trace 展示


class CostScope:
    """真实成本归因：在 run 前后快照 STATS，差量即该 run 开销。"""
    def __init__(self):
        self.cost0 = STATS["cost_usd"]
        self.calls0 = STATS["calls"]

    def result(self, answer, searches_used, events):
        return BaselineResult(
            answer=answer,
            searches_used=searches_used,
            llm_calls=STATS["calls"] - self.calls0,
            cost_usd=STATS["cost_usd"] - self.cost0,
            events=events,
        )
