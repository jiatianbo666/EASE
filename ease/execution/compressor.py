"""EvidenceCompressor —— 结构化事实抽取（构想三压缩层）。

真实小模型抽取 Fact[]；core 子问题支撑证据 preserve_full 保留全文（防 AgentDiet 陷阱）。
相似事实合并（cosine>0.9）在 agent 层按需调用，本类负责单文档结构化。
"""
from ..llm.prompts import COMPRESS
from .memory import Evidence, Fact


class EvidenceCompressor:
    def __init__(self, llm, config=None):
        self.llm = llm

    def compress(self, doc, subq, round_no=0, budget_ledger=None):
        """把检索文档转成 Evidence（结构化 Fact[]）。真实小模型调用。"""
        prompt = COMPRESS.format(subq=subq.text, title=doc.title, text=doc.text)
        result = self.llm.chat_json(
            "small",
            [{"role": "user", "content": prompt}],
            schema_hint=('{"facts":[{"ftype":"entity|relation|numeric|time|statement",'
                         '"subject":"string","predicate":"string","value":"string",'
                         '"source_sent":"string"}]}'),
            budget_ledger=budget_ledger,
        )
        facts = []
        if result and isinstance(result.get("facts"), list):
            for f in result["facts"]:
                if not isinstance(f, dict):
                    continue
                facts.append(Fact(
                    ftype=str(f.get("ftype", "statement")),
                    subject=str(f.get("subject", "")),
                    predicate=str(f.get("predicate")) or None,
                    value=str(f.get("value", "")),
                    source_sent=str(f.get("source_sent")) or None,
                ))
        return Evidence(
            eid="",  # memory.add 分配
            source_doc=doc.doc_id,
            title=doc.title,
            facts=facts,
            raw_text=doc.text,
            added_round=round_no,
        )
