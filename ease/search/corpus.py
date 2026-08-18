"""HotpotQA 语料构建与加载。

原始格式（distractor，已实测确认）：
  context[i] = [title, [sentence1, sentence2, ...]]   （list 格式）
  或 dict 格式 {"title": ..., "sentences": [...]}
  supporting_facts = [[title, sent_idx], ...]

语料 = 7405 题的 context 段落并集（按 title 去重），含 gold_for 映射。
"""
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    gold_for: dict = field(default_factory=dict)   # {qid: [sent_idx, ...]}
    source_file: str = ""

    @staticmethod
    def make_id(title):
        return "doc_" + hashlib.md5(title.encode("utf-8")).hexdigest()[:12]


def _normalize_paragraph(para):
    """list 或 dict 两种格式 → (title, [sentences])"""
    if isinstance(para, dict):
        title = para.get("title", "")
        sents = para.get("sentences", para.get("text", []))
    else:
        title = para[0]
        sents = para[1]
    if isinstance(sents, str):
        sents = [sents]
    return title, list(sents)


def _collect_sf(q):
    """supporting_facts → {title: [sent_idx, ...]}"""
    out = {}
    for t, sid in q.get("supporting_facts", []):
        out.setdefault(t, []).append(sid)
    return out


def build_docs(questions, source_file=""):
    """从原始问题列表构建去重语料 Doc[]。"""
    docs = {}
    for q in questions:
        qid = q.get("_id")
        sf = _collect_sf(q)
        for para in q.get("context", []):
            title, sents = _normalize_paragraph(para)
            if not title:
                continue
            text = " ".join(sents)
            if title not in docs:
                docs[title] = Doc(doc_id=Doc.make_id(title), title=title,
                                  text=text, gold_for={}, source_file=source_file)
            else:
                # 同名段落不同题可能略有差异：追加未见过的句子，保证 gold 句都在
                existing = docs[title].text
                extra = [s for s in sents if s not in existing]
                if extra:
                    docs[title].text = existing + " " + " ".join(extra)
            gs = sf.get(title, [])
            if gs:
                docs[title].gold_for[qid] = gs
    return list(docs.values())


def load_docs(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(Doc(**json.loads(line)))
    return docs


def save_docs(docs, path):
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
