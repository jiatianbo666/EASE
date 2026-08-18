"""WorkingMemory —— 结构化证据库 + 指纹去重 + 上下文组装（task.md §6.2 §7.7）。"""
from dataclasses import dataclass, field


@dataclass
class Fact:
    ftype: str = "statement"    # entity|relation|numeric|time|statement
    subject: str = ""
    predicate: str = None
    value: str = ""
    source_sent: str = None
    verdict: str = "fact"       # fact | claim | conflict


@dataclass
class Evidence:
    eid: str = ""
    source_doc: str = ""
    title: str = ""
    facts: list = field(default_factory=list)
    raw_text: str = ""
    added_round: int = 0
    used_in_answer: bool = False

    def compact(self, max_chars=600):
        """紧凑事实串（供小模型/大模型上下文）。"""
        lines = []
        for f in self.facts:
            if f.predicate:
                lines.append(f"[{self.eid}|{f.ftype}] {f.subject} — {f.predicate}: {f.value}")
            else:
                lines.append(f"[{self.eid}|{f.ftype}] {f.subject}: {f.value}")
        s = "\n".join(lines)
        return s[:max_chars]


class WorkingMemory:
    def __init__(self):
        self.evidence = []
        self.fingerprints = {}      # doc 指纹 -> eid（会话级缓存，防重复检索/入库）
        self._counter = 0

    def _next_eid(self):
        self._counter += 1
        return f"ev_{self._counter}"

    def seen(self, fp):
        return fp in self.fingerprints

    def add(self, evidence, fingerprint):
        """入库：分配 eid；指纹重复则返回既有 eid。"""
        if fingerprint in self.fingerprints:
            return self.fingerprints[fingerprint]
        evidence.eid = self._next_eid()
        self.evidence.append(evidence)
        self.fingerprints[fingerprint] = evidence.eid
        return evidence.eid

    def candidates_for(self, slot_text):
        """与缺口文本有词重叠的证据。"""
        toks = {w for w in slot_text.lower().split() if len(w) > 3}
        out = []
        for e in self.evidence:
            low = e.raw_text.lower()
            if any(w in low for w in toks):
                out.append(e)
        return out

    def context_text(self, max_chars=8000):
        """最终答案生成的证据上下文（结构化事实，带 eid）。"""
        chunks = []
        used = 0
        for e in self.evidence:
            c = e.compact()
            if used + len(c) > max_chars:
                break
            chunks.append(c)
            used += len(c)
        return "\n".join(chunks) if chunks else "(无证据)"

    def all_facts(self):
        return [f for e in self.evidence for f in e.facts]
