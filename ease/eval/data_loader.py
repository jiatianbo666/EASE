"""评测数据加载 —— HotpotQA dev（真实语料，与检索后端同一份）。"""
import json
import random

from ..utils.config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json"


def load_hotpot(path=None):
    """读取全部问题；每条含 _id / question / answer / supporting_facts。"""
    path = path or DEFAULT_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_questions(n, seed=42, qids=None, exclude_qids=(), path=None):
    """按 seed 采样 n 条真实问题（qids 优先；可排除已用 qid）。"""
    data = load_hotpot(path)
    if qids:
        by_id = {q["_id"]: q for q in data}
        picked = [by_id[q] for q in qids if q in by_id]
    else:
        pool = [q for q in data if q["_id"] not in set(exclude_qids)]
        rng = random.Random(seed)
        picked = rng.sample(pool, min(n, len(pool)))
    return picked


def qtype_stats(questions):
    """问题类型分布（bridge/comparison/boolean…），用真实数据统计。"""
    stats = {}
    for q in questions:
        t = q.get("_type", "unknown")
        stats[t] = stats.get(t, 0) + 1
    return stats
