"""QueryGenerator —— 面向缺口的精准查询生成（P1：检索必须绑定缺口）。

真实小模型调用；技能库高价值查询模板作 few-shot；查询指纹防重复（P1 反"再多搜搜"）。
锚点感知：传入已解析实体（hop-1 从证据中解析出的实体值），prompt 层让 LLM 自然锚定 +
纯 python 后置回退兜底 LLM 漂移（属性槽查询不含任何锚点 → 确定性追加 top anchor）。
"""
import re

from ..llm.prompts import QUERY_GEN, QUERY_GEN_ANCHORS

# 锚点注入的槽型：属性查询（数值/时间/关系）正是"无锚点 → 检索 miss"的高发槽。
# 实体槽的 slot.text 通常自带实体，不做强制注入。
_ANCHOR_SLOT_TYPES = ("numeric", "time", "relation")


class QueryGenerator:
    def __init__(self, llm, config=None):
        self.llm = llm
        self._seen = set()

    @staticmethod
    def fingerprint(query):
        q = re.sub(r"[^a-z0-9 ]", " ", query.lower())
        return re.sub(r"\s+", " ", q).strip()

    def has_seen(self, fingerprint):
        return fingerprint in self._seen

    @staticmethod
    def _contains_anchor(query, anchors):
        """query 是否已含任一 ≥2 词的锚点子串（大小写不敏感）。"""
        q = query.lower()
        for a in anchors:
            if len(a.split()) >= 2 and a.lower() in q:
                return True
        return False

    @staticmethod
    def _fill_templates(templates, resolved_entities):
        """注入前把模板里的 <entityN> 占位符用当前题已解析实体确定性填充。
        占位符 N 对应 resolved_entities[N-1]；填不出则保留占位符（prompt 说明）。
        """
        ents = list(resolved_entities or [])
        filled = []
        for t in (templates or [])[:3]:
            out = re.sub(
                r"<entity(\d+)>",
                lambda m: (ents[int(m.group(1)) - 1]
                           if int(m.group(1)) <= len(ents) else m.group(0)),
                t,
            )
            filled.append(out)
        return filled

    def generate(self, subq, slot, templates=None, resolved_entities=None,
                 budget_ledger=None):
        """返回 {"query","focus","fingerprint"}；失败返回 None。

        resolved_entities：有序已解析实体列表（best first，来自 agent 工作记忆）。
        templates：抽象化后的高价值查询骨架（<entityN> 占位符 → 当前题实体）。
        """
        few_shot = ""
        if templates:
            t = self._fill_templates(templates, resolved_entities)
            few_shot = ("参考高价值查询模板（<entityN> 是占位符，"
                        "请用当前问题的实际实体名替换；已填的用原样）：\n"
                        + "\n".join(f"- {x}" for x in t))
        anchor_block = ""
        if resolved_entities:
            anchor_block = QUERY_GEN_ANCHORS.format(
                entities="；".join(resolved_entities[:6]))
        prompt = QUERY_GEN.format(few_shot=few_shot, resolved_entities=anchor_block,
                                  slot_text=slot.text, subq=subq.text)

        result = self._ask(prompt, budget_ledger)
        if not result or not result.get("query"):
            return None
        query = result["query"].strip()
        fp = self.fingerprint(query)

        # 防重复：查询已查过 → 换角度重生成一次（针对同一槽）
        if fp in self._seen:
            result = self._ask(prompt + f"\n之前已查过类似内容，请换一个更具体的新角度，聚焦：{slot.text}", budget_ledger)
            if result and result.get("query"):
                query = result["query"].strip()
                fp = self.fingerprint(query)

        # 确定性锚点回退：属性槽查询不含任何已解析实体 → 追加 top anchor。
        # 不依赖 LLM 是否遵守"逐字嵌入实体"指令，保证 hop-2 依赖查询永不丢锚点。
        if (slot.stype in _ANCHOR_SLOT_TYPES and resolved_entities
                and not self._contains_anchor(query, resolved_entities)):
            query = f"{query} {resolved_entities[0]}"
            fp = self.fingerprint(query)

        self._seen.add(fp)
        return {"query": query, "focus": result.get("focus", ""), "fingerprint": fp}

    def _ask(self, prompt, budget_ledger):
        return self.llm.chat_json(
            "small",
            [{"role": "user", "content": prompt}],
            schema_hint='{"query":"string","focus":"string"}',
            budget_ledger=budget_ledger,
        )
