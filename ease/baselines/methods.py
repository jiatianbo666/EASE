"""四条基线实现 —— CoT / RAG-once / IRCoT / ReAct。

全部真实组件：小模型调度（deepseek-v4-flash）、大模型作答（deepseek-v4-pro）、
真实检索（与 EASE 同一 OfflineSearchTool）。无任何 mock。
"""
from .common import CostScope
from .prompts import COT_FINAL, RAG_ANSWER, IRCOT_STEP, REACT_STEP


def _context(docs, max_chars=6000):
    parts, used = [], 0
    for d in docs:
        chunk = f"[{d.rank}] {d.title}\n{d.text[:500]}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n".join(parts) or "(无检索结果)"


class CoT:
    """纯大模型推理，零检索。1 次 large 调用。"""
    name = "CoT"

    def __init__(self, llm, search=None, max_steps=None):
        self.llm = llm

    def run(self, q, **kw):
        scope = CostScope()
        r = self.llm.chat_json(
            "large",
            [{"role": "user", "content": COT_FINAL.format(question=q["question"])}],
            schema_hint='{"answer":"string"}',
        )
        answer = str(r.get("answer", "")) if r else ""
        return scope.result(answer, 0, [{"round": 1, "kind": "answer", "detail": "no retrieval"}])


class RAGOnce:
    """检索一次（问题原样，k=8）→ 大模型作答。1 次检索 + 1 次 large。"""
    name = "RAG-Once"

    def __init__(self, llm, search, max_steps=None):
        self.llm, self.search = llm, search

    def run(self, q, **kw):
        scope = CostScope()
        c0 = self.search.calls
        docs = self.search.evidence_search(q["question"], k=8, qid=q["_id"])
        r = self.llm.chat_json(
            "large",
            [{"role": "user", "content": RAG_ANSWER.format(context=_context(docs), question=q["question"])}],
            schema_hint='{"answer":"string"}',
        )
        answer = str(r.get("answer", "")) if r else ""
        return scope.result(answer, self.search.calls - c0,
                            [{"round": 1, "kind": "retrieve", "detail": q["question"][:60]}])


class IRCoT:
    """交错检索-推理：small 每轮给 thought+query，检索累积上下文，直到 query 为空或达上限。"""
    name = "IRCoT"

    def __init__(self, llm, search, max_steps=6):
        self.llm, self.search = llm, search
        self.max_steps = max_steps

    def run(self, q, **kw):
        scope = CostScope()
        c0 = self.search.calls
        docs, events, question = [], [], q["question"]
        for i in range(self.max_steps):
            r = self.llm.chat_json(
                "small",
                [{"role": "user", "content": IRCOT_STEP.format(context=_context(docs), question=question)}],
                schema_hint='{"thought":"string","query":"string"}',
            )
            if not r:
                break
            query = str(r.get("query", "")).strip()
            events.append({"round": i + 1, "kind": "thought", "detail": str(r.get("thought", ""))[:60]})
            if not query:
                break
            new = self.search.evidence_search(query, qid=q["_id"])
            docs.extend(new)
            events.append({"round": i + 1, "kind": "search", "detail": query[:60]})
        r = self.llm.chat_json(
            "large",
            [{"role": "user", "content": RAG_ANSWER.format(context=_context(docs), question=question)}],
            schema_hint='{"answer":"string"}',
        )
        answer = str(r.get("answer", "")) if r else ""
        return scope.result(answer, self.search.calls - c0, events)


class ReAct:
    """ReAct 循环：small 选行动（search/answer），真实检索，直到 answer 或达上限。"""
    name = "ReAct"

    def __init__(self, llm, search, max_steps=8):
        self.llm, self.search = llm, search
        self.max_steps = max_steps

    def run(self, q, **kw):
        scope = CostScope()
        c0 = self.search.calls
        docs, events, question = [], [], q["question"]
        answer = ""
        for i in range(self.max_steps):
            r = self.llm.chat_json(
                "small",
                [{"role": "user", "content": REACT_STEP.format(context=_context(docs), question=question)}],
                schema_hint='{"thought":"string","action":"search|answer","query":"string","answer":"string"}',
            )
            if not r:
                break
            action = str(r.get("action", "search"))
            events.append({"round": i + 1, "kind": f"react:{action}", "detail": str(r.get("thought", ""))[:60]})
            if action == "answer":
                answer = str(r.get("answer", ""))
                break
            query = str(r.get("query", "")).strip()
            if not query:
                break
            docs.extend(self.search.evidence_search(query, qid=q["_id"]))
        if not answer:
            r = self.llm.chat_json(
                "large",
                [{"role": "user", "content": RAG_ANSWER.format(context=_context(docs), question=question)}],
                schema_hint='{"answer":"string"}',
            )
            answer = str(r.get("answer", "")) if r else ""
        return scope.result(answer, self.search.calls - c0, events)


def build_baselines(llm, search, config=None):
    """实例化全部基线并包装为可调用 fn(q)（与 EASE 的 make_ease_method 一致）。
    返回 {name: fn}，harness 直接 fn(question)。"""
    c = config or {}
    instances = {
        "CoT": CoT(llm, search, c.get("max_steps", None)),
        "RAG-Once": RAGOnce(llm, search),
        "IRCoT": IRCoT(llm, search, c.get("ircot_steps", 6)),
        "ReAct": ReAct(llm, search, c.get("react_steps", 8)),
    }
    return {name: (lambda q, _inst=inst: _inst.run(q)) for name, inst in instances.items()}
