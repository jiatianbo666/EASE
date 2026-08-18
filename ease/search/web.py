"""真实网页搜索后端 —— Tavily API（需 TAVILY_API_KEY，.env 已配置）。

- 无 key 时启动即明确报错，绝不静默伪造结果。
- 评测不可复现，故不使用本后端；仅用于 demo_single.py 演示真实网页搜索。
- 调用计数/耗时与离线后端同一口径。
"""
import os
import re
import time

import requests

from .base import SearchTool, RetrievedDoc
from ..utils.config import load_env


class WebSearchTool(SearchTool):
    def __init__(self, config):
        super().__init__()
        load_env()
        self.api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        self.max_results = config.get("max_results", 5)
        self.search_depth = config.get("search_depth", "basic")
        if not self.api_key:
            raise ValueError(
                "TAVILY_API_KEY 未配置：网页搜索后端不可用。\n"
                "请在项目根 .env 填入 TAVILY_API_KEY，或改用 EASE_SEARCH_BACKEND=offline。"
            )

    def _tavily_search(self, query, max_results=None, depth=None):
        url = "https://api.tavily.com/search"
        payload = {
            "query": query,
            "max_results": max_results or self.max_results,
            "search_depth": depth or self.search_depth,
            "include_answer": False,
        }
        r = requests.post(url, json=payload,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          timeout=30)
        r.raise_for_status()
        return r.json().get("results", [])

    def evidence_search(self, query, gap_slot=None, k=None, qid=None):
        t0 = time.time()
        raw = self._tavily_search(query, max_results=k or self.max_results)
        docs = [
            RetrievedDoc(
                doc_id=r.get("url", ""),
                title=r.get("title", ""),
                text=r.get("content", ""),
                snippet=r.get("content", "")[:300],
                score=r.get("score", 0.0),
                rank=i + 1,
                is_gold=False,  # 网页后端无法判定 gold
            )
            for i, r in enumerate(raw)
        ]
        self._count(query, docs, (time.time() - t0) * 1000)
        return docs

    def fast_lookup(self, query, k=2, qid=None):
        return self.evidence_search(query, k=k, qid=qid)

    def deep_scrape(self, doc_ref):
        """doc_ref 为 URL；抓取并粗略抽取正文（真实 HTTP）。"""
        t0 = time.time()
        r = requests.get(doc_ref, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = re.sub(r"<script.*?</script>|<style.*?</style>", "", r.text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        doc = RetrievedDoc(doc_id=doc_ref, title=doc_ref, text=text[:4000],
                           snippet=text[:300], score=0.0, rank=1)
        self._count(doc_ref, [doc], (time.time() - t0) * 1000)
        return doc
