"""经验层 SkillStore —— SQLite + bge 嵌入检索 + ExpeL 投票生命周期。

设计要点（task.md §7.5）：
  - 存储：SQLite（payload JSON）+ 描述嵌入（检索用）
  - 检索：真实向量余弦 kNN，返回检索面
  - 写入：仅 verified=True（P5 无执行无记忆）；描述余弦>dedupe_threshold 视为合并更新
  - 生命周期：importance 新=2，UPVOTE+1，DOWNVOTE-1，到 0 删除
  - assert_integrity：skills 行数 == skill_emb 行数（防陈旧索引，Voyager 式一致性）
"""
import datetime
import json
import sqlite3

import numpy as np

from .skill import Skill
from ..utils.config import PROJECT_ROOT


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


class SkillStore:
    def __init__(self, config, embedder=None):
        self.db_path = PROJECT_ROOT / config.get("skills_db", "data/skills/skills.db")
        self.dedupe_threshold = config.get("dedupe_threshold", 0.95)
        self.retrieve_top_k = config.get("retrieve_top_k", 5)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if embedder is None:
            from ..embeddings.embedder import Embedder
            embedder = Embedder()
        self.embedder = embedder
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    # ---------- 存储 ----------
    def _init_schema(self):
        cur = self._conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS skills (name TEXT PRIMARY KEY, payload TEXT, updated_at TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS skill_emb (name TEXT PRIMARY KEY, embedding BLOB)")
        self._conn.commit()

    def close(self):
        self._conn.close()

    # ---------- 检索 ----------
    def retrieve(self, query_description, top_k=None, task_type=None, min_sim=0.0):
        """按任务特征描述检索技能（真实向量余弦），返回最相似 Skill[]。

        phase2 门控：传 task_type 时先按 `skill.task_type == task_type` 硬过滤
        （不同任务类型的技能在结构上不可能被注入）；min_sim 只兜底同类型内
        的低相关匹配（默认 0.0 保持旧行为）。
        """
        qv = self.embedder.embed(query_description)[0]
        rows = self._all_emb()
        if not rows:
            return []
        cands = []
        for name, blob in rows:
            if task_type is not None:
                s = self._load_skill(name)
                if not s or s.task_type != task_type:
                    continue
            emb = np.frombuffer(blob, dtype=np.float32)
            sim = float((emb @ qv).item())
            cands.append((name, sim))
        if not cands:
            return []
        cands.sort(key=lambda x: -x[1])
        k = min(top_k or self.retrieve_top_k, len(cands))
        out = []
        for name, sim in cands[:k]:
            if sim < min_sim:
                continue
            s = self._load_skill(name)
            if s:
                out.append(s)
        return out

    def _all_emb(self):
        cur = self._conn.cursor()
        cur.execute("SELECT name, embedding FROM skill_emb")
        return cur.fetchall()

    def _load_skill(self, name):
        cur = self._conn.cursor()
        cur.execute("SELECT payload FROM skills WHERE name=?", (name,))
        row = cur.fetchone()
        if not row:
            return None
        return Skill.from_dict(json.loads(row[0]))

    # ---------- 写入（P5：仅 verified） ----------
    def add_skill(self, skill, verified=True):
        """返回 "inserted" / "merged"(去重合并) / "skipped"(未验证) / "error"。
        去重：新描述与已有描述余弦 > 阈值 → 合并更新而非新增。
        """
        if not verified:
            return "skipped"
        if not skill.name or not skill.description:
            return "error"
        skill.updated_at = _now()
        new_emb = self.embedder.embed(skill.description)[0]

        for name, blob in self._all_emb():
            old_emb = np.frombuffer(blob, dtype=np.float32)
            sim = float((old_emb @ new_emb).item())
            if sim > self.dedupe_threshold:
                existing = self._load_skill(name)
                if existing is None:
                    continue
                self._merge(existing, skill)
                self._save_skill(existing)
                return "merged"
        skill.created_at = skill.created_at or _now()
        # 同名字覆写时保留已有 task_rules（对比提炼规则不能因 derive 重新生成被清掉）
        existing = self._load_skill(skill.name)
        if existing and existing.task_rules:
            for r in existing.task_rules:
                if r not in skill.task_rules:
                    skill.task_rules.append(r)
        self._save_skill(skill)
        return "inserted"

    def _merge(self, existing, new):
        """描述高度相似时合并：保留更丰富的载荷与统计。"""
        if new.decomposition_template and len(new.decomposition_template) > len(existing.decomposition_template):
            existing.decomposition_template = new.decomposition_template
        if new.budget_ratio and not existing.budget_ratio:
            existing.budget_ratio = new.budget_ratio
        if new.query_templates and not existing.query_templates:
            existing.query_templates = new.query_templates
        # 断点2修复：query_meta 是跨题复用的查询骨架，应随经验累积（去重追加，
        # 不因"已有"就丢弃新的更好骨架——否则一直学不进去新查询）。
        if new.query_meta:
            have = {(m.get("stype"), m.get("text")) for m in existing.query_meta}
            for m in new.query_meta:
                if m.get("text") and (m.get("stype"), m.get("text")) not in have:
                    existing.query_meta.append(m)
                    have.add((m.get("stype"), m.get("text")))
            existing.query_meta = existing.query_meta[:10]  # 防无限增长
        for rule in new.task_rules:
            if rule not in existing.task_rules:
                existing.task_rules.append(rule)
        for lesson in new.failure_lessons:
            if lesson not in existing.failure_lessons:
                existing.failure_lessons.append(lesson)
        existing.updated_at = _now()

    def save(self, skill):
        """直接按 name 覆写（不做去重合并），用于写 task_rules 等元数据更新。"""
        skill.updated_at = _now()
        self._save_skill(skill)

    def _save_skill(self, skill):
        emb = self.embedder.embed(skill.description)[0].astype(np.float32)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO skills (name, payload, updated_at) VALUES (?,?,?)",
            (skill.name, json.dumps(skill.to_dict(), ensure_ascii=False), skill.updated_at))
        cur.execute(
            "INSERT OR REPLACE INTO skill_emb (name, embedding) VALUES (?,?)",
            (skill.name, emb.tobytes()))
        self._conn.commit()

    # ---------- 生命周期 ----------
    def upvote(self, name):
        skill = self._load_skill(name)
        if not skill:
            return False
        skill.importance += 1
        skill.usage_count += 1
        skill.updated_at = _now()
        self._save_skill(skill)
        return True

    def downvote(self, name, reason=""):
        """返回 True=仍在库中 / "deleted"=已删除 / False=不存在。"""
        skill = self._load_skill(name)
        if not skill:
            return False
        skill.importance -= 1
        if reason and reason not in skill.failure_lessons:
            skill.failure_lessons.append(reason)
        skill.updated_at = _now()
        if skill.importance <= 0:
            self.delete(name)
            return "deleted"
        self._save_skill(skill)
        return True

    def delete(self, name):
        cur = self._conn.cursor()
        cur.execute("DELETE FROM skills WHERE name=?", (name,))
        cur.execute("DELETE FROM skill_emb WHERE name=?", (name,))
        self._conn.commit()

    def record_usage(self, name, success, cost):
        """ExpeL 统计：EMA 更新成功率与平均成本（在复用技能后调用）。"""
        skill = self._load_skill(name)
        if not skill:
            return
        n = skill.usage_count + 1
        skill.success_rate = (skill.success_rate * skill.usage_count + (1.0 if success else 0.0)) / n
        skill.avg_cost = (skill.avg_cost * skill.usage_count + cost) / n
        skill.usage_count = n
        skill.updated_at = _now()
        self._save_skill(skill)

    def get(self, name):
        return self._load_skill(name)

    def all_skills(self):
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM skills")
        return [self._load_skill(r[0]) for r in cur.fetchall()]

    def count(self):
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM skills")
        return cur.fetchone()[0]

    def assert_integrity(self):
        """技能行数 == 嵌入行数（防陈旧索引）。"""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM skills")
        n1 = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM skill_emb")
        n2 = cur.fetchone()[0]
        return n1 == n2
