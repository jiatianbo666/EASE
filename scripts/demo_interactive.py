"""交互式 CLI demo —— 输入任意问题，EASE 实时作答（真实 API + 真实检索）。

用法: python scripts/demo_interactive.py            # 离线语料后端
      python scripts/demo_interactive.py --web      # Tavily 网页后端
输入 exit 退出。
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config
from ease.llm.client import LLMClient
from ease.embeddings.embedder import Embedder
from ease.experience.skill_store import SkillStore
from ease.agent.ease_agent import EASEAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", action="store_true", help="使用 Tavily 网页搜索后端")
    args = ap.parse_args()

    cfg = load_config()
    embedder = Embedder()
    llm = LLMClient(cfg["llm"])
    if args.web:
        from ease.search.web import WebSearchTool
        search = WebSearchTool(cfg["search"]["web"])
        backend = "Tavily 网页搜索"
    else:
        from ease.search.offline import OfflineSearchTool
        search = OfflineSearchTool(cfg["search"]["offline"], embedder=embedder)
        backend = "HotpotQA 离线语料"
    store = SkillStore(cfg["experience"], embedder=embedder)
    agent = EASEAgent(cfg, llm, search, store, embedder=embedder)

    print(f"EASE 交互 demo · 后端: {backend} · 技能库 {store.count()} 条")
    print("输入问题（exit 退出）:")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        result = agent.run(question)
        t = result.trace
        print(f"\n  [答案] {result.answer[:500]}")
        print(f"  [开销] {t.budget_used_searches} 次检索 · {t.llm_calls} 次 LLM · "
              f"${t.cost_usd:.4f}")


if __name__ == "__main__":
    main()
