"""成本核算 —— DeepSeek 分时定价（官方价格，2026-08-17 抓取验证）。

来源：https://api-docs.deepseek.com/quick_start/pricing（与第三方 chat-deep.ai 交叉一致）
- 模型版本：deepseek-v4-flash → DeepSeek-V4-Flash-0731；deepseek-v4-pro → DeepSeek-V4-Pro-0813
- 单位：USD / 每 1M tokens
- Peak 时段：01:00-04:00 与 06:00-10:00 UTC；其余时间为 off-peak（off-peak = peak 一半）
- pricing_mode: auto=按当前时段自动选价 | peak=恒用 peak | off-peak=恒用 off-peak

价格随时可能调整；若变动请更新本表并注明来源日期。
"""
import datetime

# (off_peak, peak) 单位：USD / 1M tokens
PRICING = {
    "deepseek-v4-flash": {
        "input_cache_hit": (0.007, 0.014),
        "input_cache_miss": (0.22, 0.44),
        "output": (0.66, 1.32),
    },
    "deepseek-v4-pro": {
        "input_cache_hit": (0.022, 0.044),
        "input_cache_miss": (0.66, 1.32),
        "output": (1.98, 3.96),
    },
}

UNITS = 1_000_000


def is_peak_hour(now_utc=None):
    """Peak 时段：01:00-04:00 与 06:00-10:00 UTC。"""
    if now_utc is None:
        now_utc = datetime.datetime.utcnow()
    h = now_utc.hour
    return (1 <= h < 4) or (6 <= h < 10)


def rates(model, mode="auto"):
    """返回 {input_cache_hit, input_cache_miss, output, peak}；模型未定价返回 None。"""
    if model not in PRICING:
        return None
    p = PRICING[model]
    if mode == "auto":
        idx = 1 if is_peak_hour() else 0
    elif mode == "peak":
        idx = 1
    else:  # off-peak
        idx = 0
    return {
        "input_cache_hit": p["input_cache_hit"][idx],
        "input_cache_miss": p["input_cache_miss"][idx],
        "output": p["output"][idx],
        "peak": idx == 1,
    }


def estimate_cost(model, usage, mode="auto"):
    """按 usage 折算美元成本。

    usage: {prompt_tokens, completion_tokens, prompt_cache_hit_tokens, ...}
    模型未定价时返回 None（调用方改用原始 token 数作为模型无关代理）。
    """
    if model not in PRICING:
        return None
    if not usage:
        return 0.0
    r = rates(model, mode)
    in_total = usage.get("prompt_tokens", 0) or 0
    out = usage.get("completion_tokens", 0) or 0
    cache_hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    cache_miss = max(0, in_total - cache_hit)
    return (
        cache_hit / UNITS * r["input_cache_hit"]
        + cache_miss / UNITS * r["input_cache_miss"]
        + out / UNITS * r["output"]
    )
