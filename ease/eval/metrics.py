"""答案指标（HotpotQA 口径）与效率指标。

- EM：正则化后精确匹配（小写、去冠词、去标点）
- F1：词级 token 精确率/召回率/F1
- 效率：searches / llm_calls / cost / rounds
全部为确定性计算，无任何模型参与。
"""
import re


def normalize(s):
    """HotpotQA 官方正则化：小写、去冠词、去标点、合并空白。"""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def em_f1(pred, gold_answers):
    """对 gold 列表取最优；返回 (em, f1)。gold_answers 可为 str 或 list。"""
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    np_, ng = normalize(pred), [normalize(g) for g in gold_answers if g]
    if not ng:
        return 0.0, 0.0
    best_em, best_f1 = 0.0, 0.0
    for g in ng:
        best_em = max(best_em, 1.0 if np_ == g else 0.0)
        best_f1 = max(best_f1, token_f1(np_.split(), g.split()))
    return best_em, best_f1


def answer_success(pred, gold_answers, f1_threshold=1.0):
    """成功判定：EM>0 或 F1>=阈值（用于技能固化门控）。"""
    e, f = em_f1(pred, gold_answers)
    return (e > 0) or (f >= f1_threshold)
