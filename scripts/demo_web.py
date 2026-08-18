"""真实网页搜索 demo —— EASE 走 Tavily 在线检索（非评测用，结果不可复现）。

用法: python scripts/demo_web.py "Who directed the 2020 film Tenet?"
（需 .env 已配置 TAVILY_API_KEY）
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config
from ease.llm.client import LLMClient
from ease.embeddings.embedder import Embedder
from ease.search.web import WebSearchTool
from ease.experience.skill_store import SkillStore
from ease.agent.ease_agent import EASEAgent


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else (
        "Who won the 2026 FIFA World Cup?")
    cfg = load_config()
    embedder = Embedder()
    llm = LLMClient(cfg["llm"])
    search = WebSearchTool(cfg["search"]["web"])     # 真实 Tavily
    store = SkillStore(cfg["experience"], embedder=embedder)
    agent = EASEAgent(cfg, llm, search, store, embedder=embedder)

    print(f"后端: Tavily 网页搜索（max_results={cfg['search']['web']['max_results']}）")
    print(f"问题: {question}\n")
    result = agent.run(question)
    t = result.trace

    print(f"[特征] {t.task_feature}  命中技能: {t.prior_skill or '(无)'}")
    for s in t.sub_questions:
        print(f"  [{s['qtype']}] {s['text']}")
    print(f"\n[执行] {len(t.searches)} 次真实网页检索（预算用 {t.budget_used_searches}）:")
    for s in t.searches:
        print(f"  Q: {s['query'][:60]}")
        for title in s["kept"]:
            print(f"    → {title[:70]}")
    print(f"  覆盖度: {t.coverage_hist}")
    print(f"\n[答案] {result.answer[:600]}")
    if result.cites:
        print(f"  引用: {result.cites}")
    print(f"\nLLM 调用 {t.llm_calls} · 成本 ${t.cost_usd:.4f}")


if __name__ == "__main__":
    main()
