"""最终答案生成（大模型 deepseek-v4-pro，推理开启；task.md §7.4 §12）。

输入：结构化证据上下文 + 元认知摘要 → 带 [eid] 引用的答案。
- 证据优先：证据部分覆盖也作答，仅零相关证据才 Unknown（实测拒绝式回答吃掉 EM）。
- 拒绝式标记命中时做一次重试（证据优先 follow-up）。
- 多候选答案（"X and Y"）时强制单提交：一次专注消歧调用 + 证据支持度确定性 tie-break。
- 大模型是唯一允许推理的环节（非对称分工）。
"""
import re

from ..llm.prompts import ANSWER_FINAL, ANSWER_RETRY, ANSWER_COMMIT

# 拒绝式标记：命中即重试。含中英文（HotpotQA 全英文，但防御性覆盖）。
_HEDGE_RE = re.compile(
    r"^\s*(unknown|uncertain|none|no common actor|"
    r"not enough( evidence| information| data)?|insufficient( evidence| information)?|"
    r"cannot determine|can't determine|cannot answer|unable to answer|could not determine)\s*[\.!]*\s*$"
    r"|^\s*(无法确定|证据不足|无法回答|不知道|不清楚|未知|不确定|没有相关信息)\s*$",
    re.IGNORECASE,
)

# 多候选连接词：切分出 ≥2 个候选实体片段（Fix B 确定性单提交）。
_MULTI_SPLIT = re.compile(r"\s+(?:and|&|/|、|及)\s+")


def _is_hedge(answer):
    return bool(answer) and bool(_HEDGE_RE.match(answer.strip()))


def _split_candidates(answer):
    """多候选列表检测：按连接词切分，各片段首字母大写、2-5 词、整串较短。

    规避误报：'Harry Potter and the Chamber of Secrets' 第二段小写开头 → 不触发；
    'Kevin Spacey and Annette Bening' 两段均大写 → 触发。返回候选列表，否则 []。
    """
    if not answer or len(answer) > 60:
        return []
    parts = _MULTI_SPLIT.split(answer.strip())
    if len(parts) < 2:
        return []
    cands = [p.strip(" .,;\"'()[]") for p in parts]
    if not all(c and c[0].isupper() for c in cands):
        return []
    if not all(1 <= len(c.split()) <= 5 for c in cands):
        return []
    return cands


def _is_multicandidate(answer):
    return bool(_split_candidates(answer))


_STOP = set(
    "the a an of and or to in on at for from by with as is are was were be been being "
    "this that these those it its he she they we you i me my your his her him them "
    "who what which where when how why not no".split()
)


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _pick_by_evidence(candidates, memory, question=""):
    """确定性 tie-break：证据出现次数 + 与问题限定词内容词的共现加分。

    共现加权让限定词命中候选时更受青睐（如问题含 Mexican/American → 证据里
    Salma Hayek 的 Mexican-American 事实加权，选 Hayek 而非西班牙裔 Penélope Cruz）。
    平局取列表第一个。纯 python，无 LLM。"""
    texts = []
    for e in memory.evidence:
        texts.append(e.raw_text)
        for f in e.facts:
            texts.append(f"{f.subject} {f.predicate or ''} {f.value}")
    blob = " ".join(texts).lower()
    q_toks = [t for t in re.findall(r"[a-z']+", question.lower())
              if t not in _STOP and len(t) > 2]
    best, best_score = candidates[0], -1
    for c in candidates:
        cl = c.lower()
        occ = blob.count(cl)
        qual = 0
        for t in texts:
            low = t.lower()
            if cl in low:
                qual += sum(1 for qt in q_toks if qt in low)
        score = occ + 2 * qual
        if score > best_score:
            best, best_score = c, score
    return best


def state_summary(state):
    """元认知摘要：覆盖度 + 每子问题状态 + 未决冲突。"""
    lines = [
        f"core_coverage={state.core_coverage():.2f} · total_coverage={state.total_coverage():.2f}"
    ]
    for sq in state.sub_questions:
        lines.append(f"- [{sq.qtype}] {sq.text} | status={sq.status} cov={sq.coverage:.2f}")
    if state.conflicts:
        lines.append(f"conflicts: {len(state.conflicts)} 处未决矛盾")
    return "\n".join(lines)


def _call(llm, prompt, budget_ledger):
    return llm.chat_json(
        "large",
        [{"role": "user", "content": prompt}],
        schema_hint='{"answer":"string","cites":["eid"],"unaddressed":["string"]}',
        budget_ledger=budget_ledger,
        temperature=llm.answer_temperature,
    )


def generate_answer(llm, state, memory, question, budget_ledger=None):
    """真实大模型调用，返回 {"answer","cites","unaddressed"}。
    拒绝式回答（Unknown/None 等）触发一次证据优先重试。"""
    prompt = ANSWER_FINAL.format(
        question=question,
        evidence_context=memory.context_text(max_chars=8000),
        state_summary=state_summary(state),
    )
    r = _call(llm, prompt, budget_ledger)
    if not r:
        return {"answer": "", "cites": [], "unaddressed": ["答案生成调用失败"]}

    answer = str(r.get("answer", "")).strip()
    if _is_hedge(answer):
        r2 = _call(llm, ANSWER_RETRY.format(
            prev=answer,
            question=question,
            evidence_context=memory.context_text(max_chars=8000),
            state_summary=state_summary(state),
        ), budget_ledger)
        if r2:
            a2 = str(r2.get("answer", "")).strip()
            if a2 and not _is_hedge(a2):
                answer = a2
                r = r2

    # 多候选 → 强制单提交（Fix B）：一次专注消歧调用。
    # 裁决原则（防消歧调用忽略问题限定词、把列表选成错误单候选）：
    #   - 消歧返回列表/拒绝式        → 用限定词感知的确定性评分（证据次数+限定词共现）
    #   - 消歧返回列表内某个候选      → 仍用确定性评分裁决（消歧可能限定词盲）
    #   - 消歧返回**新**候选（如全名）→ 信任（这是超出候选列表的细化）
    cands = _split_candidates(answer)
    if cands:
        r3 = _call(llm, ANSWER_COMMIT.format(
            candidates=answer,
            question=question,
            evidence_context=memory.context_text(max_chars=8000),
            state_summary=state_summary(state),
        ), budget_ledger)
        a3 = str(r3.get("answer", "")).strip() if r3 else ""
        if a3 and not _is_hedge(a3) and not _is_multicandidate(a3) \
                and not any(_norm(a3) == _norm(c) for c in cands):
            answer = a3
            r = r3
        else:
            answer = _pick_by_evidence(cands, memory, question)

    return {
        "answer": answer,
        "cites": [str(c) for c in (r.get("cites") or [])],
        "unaddressed": [str(u) for u in (r.get("unaddressed") or [])],
    }
