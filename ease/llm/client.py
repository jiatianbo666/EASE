"""EASE LLM 客户端 —— 真实 DeepSeek API 调用（OpenAI 兼容）。

非对称模型分工：
  small (deepseek-v4-flash)：调度 / 评估 / 压缩，默认关闭推理（thinking disabled）省钱
  large (deepseek-v4-pro)  ：最终答案生成，默认开启推理保质量

仅真实 API 调用，无任何 mock。token / 成本记账来自 API 返回的 usage。
预算账本（budget_ledger）按 P2 注入每一步：searches used X/10 · tokens Y · cost $Z · remaining R。
"""
import random
import time
from dataclasses import dataclass, field

import openai
from openai import OpenAI

from .costs import estimate_cost
from ..utils.json_utils import parse_json_tolerant

# 会话级统计（评测运行时按 run 重置）
STATS = {
    "calls": 0,
    "tokens_in": 0,
    "tokens_out": 0,
    "tokens_reasoning": 0,
    "tokens_cache_hit": 0,
    "cost_usd": 0.0,
    "calls_by_model": {},
    "errors": 0,
}


def reset_stats():
    for k in STATS:
        if isinstance(STATS[k], dict):
            STATS[k] = {}
        else:
            STATS[k] = 0


@dataclass
class ChatResult:
    content: str = ""
    reasoning_content: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    cost_usd: float = None  # None 表示该模型未定价（用 token 数代理）
    ok: bool = False
    error: str = ""
    latency_ms: float = 0.0


class LLMClient:
    def __init__(self, config: dict):
        self.base_url = config["base_url"].rstrip("/")
        self.api_key = config["api_key"]
        self.small_model = config["small_model"]
        self.large_model = config["large_model"]
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.pricing_mode = config.get("pricing_mode", "auto")
        self.small_disable_reasoning = config.get("small_disable_reasoning", True)
        self.large_disable_reasoning = config.get("large_disable_reasoning", False)
        # 最终答案生成温度：低温度压单次生成的方差（同一证据不同抽签式答案）。
        # 大模型仅用于答案生成，降温不影响调度/检索环节。
        self.answer_temperature = config.get("answer_temperature", 0.1)
        if not self.api_key:
            raise ValueError("LLMClient: api_key 为空（检查 .env 的 DEEPSEEK_API_KEY）")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    # ---------- 模型映射 ----------
    def resolve_model(self, role: str) -> str:
        if role == "small":
            return self.small_model
        if role == "large":
            return self.large_model
        raise ValueError(f"未知 role: {role!r}（应为 small/large）")

    # ---------- 主调用 ----------
    def chat(self, role, messages, *, json_mode=False, max_tokens=None, temperature=None,
             budget_ledger=None, disable_reasoning=None, tools=None) -> ChatResult:
        model = self.resolve_model(role)
        if disable_reasoning is None:
            disable_reasoning = (self.small_disable_reasoning if role == "small"
                                 else self.large_disable_reasoning)

        msgs = list(messages)
        # 语言约束（真实评测语料为英文，所有生成的 query/slot/sub-question/fact/answer
        # 必须用英文才可能与语料和 gold answer 匹配 —— 实测中文指令会让小模型输出中文查询）
        msgs = [{"role": "system",
                 "content": "You must respond in English. All generated text — search queries, "
                            "gap slots, sub-questions, extracted facts, and answers — MUST be in "
                            "English. Never output Chinese or any other language."}] + msgs
        if json_mode:
            # DeepSeek json_object 模式要求 prompt 中出现 "json" 字样
            msgs = [{"role": "system",
                     "content": "You are a precise assistant. Output ONLY a valid JSON object. "
                                "No markdown fences, no commentary, no trailing text."}] + msgs
        if budget_ledger:
            msgs = [{"role": "system", "content": f"[BUDGET LEDGER] {budget_ledger}"}] + msgs

        last_err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                kwargs = dict(model=model, messages=msgs, stream=False)
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if tools:
                    kwargs["tools"] = tools
                if disable_reasoning:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                resp = self.client.chat.completions.create(**kwargs)
                return self._to_result(resp, model, time.time() - t0)
            except Exception as e:
                last_err = e
                if not self._is_retryable(e):
                    break
                time.sleep(self._backoff(attempt, e))
        STATS["errors"] += 1
        return ChatResult(ok=False, error=f"call failed after {self.max_retries} attempt(s): {last_err}")

    # ---------- 结构化 JSON 调用（真实调用 + 容错解析 + 一次纠错重试） ----------
    def chat_json(self, role, messages, *, schema_hint=None, budget_ledger=None,
                  max_tokens=None, temperature=None, disable_reasoning=None):
        msgs = list(messages)
        if schema_hint:
            msgs = [{"role": "system",
                     "content": "You must output a JSON object conforming to this schema: "
                                + schema_hint}] + msgs
        r = self.chat(role, msgs, json_mode=True, max_tokens=max_tokens,
                      temperature=temperature, budget_ledger=budget_ledger,
                      disable_reasoning=disable_reasoning)
        if not r.ok:
            return None
        obj = parse_json_tolerant(r.content)
        if obj is not None:
            return obj
        # 一次纠错重试（真实调用，计入成本）
        retry_msgs = msgs + [
            {"role": "assistant", "content": r.content},
            {"role": "user",
             "content": "Your previous output was not valid JSON. "
                        "Output ONLY a valid JSON object, no markdown fences."},
        ]
        r2 = self.chat(role, retry_msgs, json_mode=True, max_tokens=max_tokens,
                       temperature=temperature, budget_ledger=budget_ledger,
                       disable_reasoning=disable_reasoning)
        if not r2.ok:
            return None
        return parse_json_tolerant(r2.content)

    # ---------- 内部 ----------
    def _to_result(self, resp, model, latency_ms):
        msg = resp.choices[0].message
        u = {}
        usage = resp.usage
        if usage is not None:
            ctd = getattr(usage, "completion_tokens_details", None)
            ptd = getattr(usage, "prompt_tokens_details", None)
            # 注：DeepSeek 的 completion_tokens 已包含推理 token（实测关 thinking 后
            # completion_tokens 22→1）。reasoning_tokens 明细字段有时不填充（返回 0），
            # 但按 completion_tokens 折算成本已准确，无需依赖该明细。
            reasoning = getattr(ctd, "reasoning_tokens", 0) or 0
            cache_hit = getattr(ptd, "cached_tokens", 0) or 0
            u = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                "reasoning_tokens": reasoning,
                "prompt_cache_hit_tokens": cache_hit,
            }
        cost = estimate_cost(model, u, mode=self.pricing_mode) if u else None

        STATS["calls"] += 1
        STATS["tokens_in"] += u.get("prompt_tokens", 0)
        STATS["tokens_out"] += u.get("completion_tokens", 0)
        STATS["tokens_reasoning"] += u.get("reasoning_tokens", 0)
        STATS["tokens_cache_hit"] += u.get("prompt_cache_hit_tokens", 0)
        if cost is not None:
            STATS["cost_usd"] += cost
        STATS["calls_by_model"][model] = STATS["calls_by_model"].get(model, 0) + 1

        return ChatResult(
            content=msg.content or "",
            reasoning_content=getattr(msg, "reasoning_content", "") or "",
            model=model, usage=u, cost_usd=cost, ok=True, latency_ms=latency_ms,
        )

    def _is_retryable(self, err):
        if isinstance(err, openai.APIConnectionError):
            return True
        if isinstance(err, openai.RateLimitError):
            return True
        if isinstance(err, openai.InternalServerError):
            return True
        status = getattr(err, "status_code", None)
        if status and status >= 500:
            return True
        return False

    def _backoff(self, attempt, err):
        retry_after = None
        resp = getattr(err, "response", None)
        if resp is not None and hasattr(resp, "headers"):
            ra = resp.headers.get("retry-after")
            if ra and ra.isdigit():
                retry_after = float(ra)
        base = min(2 ** attempt * 0.5, 8.0)
        jitter = random.uniform(0, 0.25)
        return retry_after if retry_after is not None else (base + jitter)
