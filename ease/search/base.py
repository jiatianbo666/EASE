"""检索后端统一接口。

信息密度保证（构想三）：检索暴露为三个原子工具 ——
  fast_lookup   低 token 快速查询（k 固定很小）
  evidence_search 针对缺口证据检索（k 自适应，节省 token）
  deep_scrape   深度读取单个文档全文（需预算授权，计 1 次检索）
每次调用都会记入 calls / call_log，供预算账本与评测使用。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    snippet: str
    score: float
    rank: int
    is_gold: bool = False   # 是否为当前问题 gold 证据（离线后端可判；网页后端不可判）


class SearchTool(ABC):
    def __init__(self):
        self.calls = 0
        self.call_log = []   # [{"query","took_ms","n","gold_hits"}] 供 trace/评测

    def _count(self, query, results, took_ms, gold_hits=0):
        self.calls += 1
        self.call_log.append({
            "query": query,
            "took_ms": round(took_ms, 1),
            "n": len(results),
            "gold_hits": gold_hits,
        })

    @abstractmethod
    def fast_lookup(self, query, k=2, qid=None):
        """低 token：仅返回最相关 k 条紧凑结果。"""

    @abstractmethod
    def evidence_search(self, query, gap_slot=None, k=None, qid=None):
        """针对缺口的证据检索：k 自适应（adaptive-k）。"""

    @abstractmethod
    def deep_scrape(self, doc_ref):
        """深度读取单个文档全文（计 1 次检索）。"""
