"""M1 真实验证 —— DeepSeek 双模型冒烟测试（真实 API 调用，无 mock）。

覆盖：
  1) 小模型 deepseek-v4-flash：thinking 关闭 → 结构化 JSON 输出
  2) 小模型：预算账本注入 + token/成本记账
  3) 大模型 deepseek-v4-pro：thinking 开启 → reasoning_content 非空

通过条件：
  - 三个真实调用全部 ok
  - 小模型 reasoning_content 为空 / reasoning_tokens = 0（thinking 已关）
  - 大模型 reasoning_content 非空（thinking 开启）
  - JSON 可解析、usage 全量记录、成本核算输出数值
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config
from ease.llm.client import LLMClient, STATS, reset_stats


def main():
    reset_stats()
    cfg = load_config()
    client = LLMClient(cfg["llm"])
    print("config: small=%s large=%s pricing_mode=%s"
          % (cfg["llm"]["small_model"], cfg["llm"]["large_model"], cfg["llm"]["pricing_mode"]))
    print("=" * 60)

    # --- 1) 小模型 JSON 输出（关推理） ---
    print("[1] small model · JSON output (thinking disabled)")
    r = client.chat_json(
        "small",
        [{"role": "user",
          "content": "把这句话转成 JSON：任务类型是桥接问题，复杂度高，涉及实体 A 与 B。"
                     "输出 JSON 对象，字段：task_type, complexity, entities(数组)。"}],
        schema_hint='{"task_type":"string","complexity":"string","entities":["string"]}',
        max_tokens=512,
    )
    assert r is not None, "small JSON 解析失败（chat_json 返回 None）"
    print("  parsed:", json.dumps(r, ensure_ascii=False)[:200])
    assert "task_type" in r, "JSON 缺 task_type 字段"

    # --- 2) 小模型文本 + 预算账本注入 + 记账 ---
    print("\n[2] small model · text + budget ledger + accounting")
    r2 = client.chat(
        "small",
        [{"role": "user", "content": "用一句话介绍你作为调度模型的职责。"}],
        budget_ledger="searches used 0/10 · tokens 0 · cost $0.0000 · remaining 10",
        max_tokens=256,
    )
    assert r2.ok, f"small text 调用失败: {r2.error}"
    print("  reasoning_content len:", len(r2.reasoning_content), "(应为 0：thinking 已关)")
    print("  reasoning_tokens:", r2.usage.get("reasoning_tokens"), "(应为 0)")
    print("  usage:", r2.usage)
    print("  cost_usd: %.6f" % (r2.cost_usd or 0.0))
    assert r2.usage["completion_tokens"] > 0, "completion_tokens 异常"
    assert r2.usage["reasoning_tokens"] == 0, "小模型 reasoning 未关闭"

    # --- 3) 大模型开推理 ---
    print("\n[3] large model · text (thinking enabled)")
    r3 = client.chat(
        "large",
        [{"role": "user",
          "content": "请用一两句话解释：为什么多跳问答任务需要显式规划检索路径，而不是一次检索到底？"}],
        max_tokens=512,
    )
    assert r3.ok, f"large 调用失败: {r3.error}"
    print("  reasoning_content len:", len(r3.reasoning_content), "(应 > 0：thinking 开启)")
    print("  reasoning_tokens:", r3.usage.get("reasoning_tokens"))
    print("  usage:", r3.usage)
    print("  cost_usd: %.6f" % (r3.cost_usd or 0.0))
    print("  answer:", r3.content[:120].replace("\n", " "))
    assert len(r3.reasoning_content) > 0, "大模型 reasoning 未开启"

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("STATS:", json.dumps(STATS, ensure_ascii=False, indent=2))
    assert STATS["calls"] >= 3
    assert STATS["tokens_in"] > 0 and STATS["tokens_out"] > 0
    assert STATS["cost_usd"] > 0, "成本核算未输出数值"
    print("\n✅ ALL SMOKE TESTS PASSED (real API · real usage · real cost)")


if __name__ == "__main__":
    main()
