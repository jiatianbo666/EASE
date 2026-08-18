"""断点3：持久化 episode 池 + 周期增量对比提炼。

设计（PROJECT_JOURNAL §6，方案 A 落地）：
  - episode 池是 build_contrast_rules 的前置条件——成败对比需要同类型的
    成功+失败轨迹对。
  - 每轮评测（warm 预训练 + warm 评测）结束，把新 run 追加进池（按 qid 去重），
    然后只对"有新 episode 到账的 task_type"跑 build_contrast_rules。
  - build_contrast_rules 天然按类型过滤：只有该类型"同时有成功+失败"才调用 LLM
    （每类型每批次一次调用 ~$0.001），只有成功走确定性模板（零 LLM），
    所以"周期跑"不会烧钱，无需额外的触发信号。
  - 持久化：episodes.jsonl（逐行一条 episode dict，append-only）+ .meta.json
    （记录每类型已提炼到的 episode 数，用于增量判定）。
  防泄漏：episode_to_dict 不含金标/评测答案，只含问题文本、分解、查询、停止轮、
  覆盖度、成败标签——这些是"经验"不是"答案"。
"""
import json
from pathlib import Path

from .extractor import episode_to_dict, episode_from_dict


class EpisodePool:
    def __init__(self, path):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta")
        self._meta = {}
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self._meta = {}
        self._pool = self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if not self.path.exists():
            return []
        out = []
        seen = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue  # 半行写入等损坏行：跳过
            qid = d.get("qid")
            if not qid or qid in seen:
                continue
            seen.add(qid)
            out.append(episode_from_dict(d))
        return out

    def add(self, runs):
        """追加新 episode（按 qid 去重），返回实际新增数。"""
        have = {e.qid for e in self._pool}
        fresh = []
        for r in runs or []:
            if r is None or r.qid in have:
                continue
            have.add(r.qid)
            fresh.append(r)
        if not fresh:
            return 0
        self._pool.extend(fresh)
        with open(self.path, "a", encoding="utf-8") as f:
            for r in fresh:
                f.write(json.dumps(episode_to_dict(r), ensure_ascii=False) + "\n")
        return len(fresh)

    def load(self):
        return list(self._pool)

    def count(self):
        return len(self._pool)

    # ---------- 周期增量提炼 ----------
    def refresh_contrast(self, store, llm, budget_ledger=None):
        """只对"新增 episode 到账的 task_type"跑 build_contrast_rules，写入技能 task_rules。

        返回统计 dict：{types, rules, llm_calls, episodes}。
        """
        from .extractor import build_contrast_rules
        by_type = {}
        for e in self._pool:
            by_type.setdefault(e.task_type, []).append(e)
        stats = {"types": 0, "rules": 0, "llm_calls": 0, "episodes": len(self._pool)}
        for tt, eps in by_type.items():
            prev = self._meta.get(tt, 0)
            if len(eps) <= prev:
                continue  # 该类型无新 episode → 跳过（增量，不重复烧 LLM）
            # LLM 调用仅发生在"同类型同时有成功+失败"时（与 build_contrast_rules 内部一致）
            if any(e.success for e in eps) and any(not e.success for e in eps):
                stats["llm_calls"] += 1
            rules_map = build_contrast_rules(eps, llm, budget_ledger)
            rules = rules_map.get(tt, [])
            sk = store.get(f"skill-{tt}")
            if sk and rules:
                sk.task_rules = rules
                store.save(sk)
                stats["rules"] += len(rules)
                stats["types"] += 1
            self._meta[tt] = len(eps)  # 标记已提炼到该数量
        self._save_meta()
        return stats

    def _save_meta(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
