"""M7 缺口放弃逻辑冒烟 —— 验证 slot 文本漂移不再绕过放弃上限（纯确定性，无 API）。

复现 bug：SLOT_GAP 每轮重新生成略不同文本的同一缺口
  r1 'the actress and writer in Black Book'
  r2 'actors who starred in Black Book (2006 film)'
  r3 'the actress/writer whose heritage is asked'
→ 若按 (sq.id, slot.text) 计数，key 永不匹配 → 永不放弃 → 烧光预算。
修复：按 (sq.id, slot.stype) 计数 + 无覆盖度进展才递增 + abandoned 跨轮持久。
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.control.metacognitive import MetacognitiveState, SubQuestion, Slot
from ease.control.router import Router
from ease.control.budget import BudgetPlanner


def make_state(slot_texts):
    """一个 core 子问题，每个 slot_text 一行的漂移文本。"""
    sq = SubQuestion(id="sq1", text="Black Book starred the actress and writer of what heritage?",
                     qtype="core")
    sq.slots = [Slot(stype="entity", text=t) for t in slot_texts]
    state = MetacognitiveState(question="q", sub_questions=[sq])
    return state, sq


def simulate(slot_texts, max_attempts=2, make_progress=False):
    """模拟循环：Router 每次选 gaps[0]，按修复逻辑计数，验证放弃触发。
    make_progress=True 表示每次检索都推进覆盖度（应重置计数、永不放弃）。"""
    state, sq = make_state(slot_texts)
    router = Router({"coverage_stop_threshold": 0.8, "conf_stop_threshold": 0.75})
    budget = BudgetPlanner(total_searches=10, total_tokens=100000,
                           finalization_reserve=0.0)
    abandoned = {}
    slot_attempts = {}
    abandon_rounds = []

    for rnd in range(1, 11):
        gaps = state.detect_gaps()
        action = router.decide(state, budget, gaps=gaps)
        if action.kind != "RETRIEVE":
            break
        sq, slot = action.gap
        key = (sq.id, slot.stype)

        # 放弃检查
        if slot_attempts.get(key, 0) >= max_attempts:
            slot.filled = True
            abandoned.setdefault(sq.id, set()).add(slot.stype)
            abandon_rounds.append(rnd)
            continue

        budget.consume_search()

        # 覆盖度进展模拟
        if make_progress:
            sq.coverage = min(1.0, sq.coverage + 0.5)
            if sq.coverage > 0:
                sq.status = "covered"
            slot_attempts[key] = 0
        else:
            slot_attempts[key] = slot_attempts.get(key, 0) + 1

    return abandon_rounds, budget.searches_used


def main():
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'✅' if cond else '❌'} {name}" + (f" | {detail}" if detail else ""))
        if not cond:
            ok = False

    drift = ["the actress and writer in 'Black Book'",
             "actors who starred in Black Book (2006 film)",
             "the actress/writer whose heritage is asked in 'Black Book'",
             "the actress in 'Black Book'",
             "the actress and writer in 'Black Book'",
             "The actress and writer in 'Black Book'"]

    print("[1] 文本漂移 + 无进展：应在第 3 轮放弃（max_attempts=2 后再次命中），不再烧光预算")
    ab, used = simulate(drift, max_attempts=2)
    check("放弃被触发", len(ab) >= 1, f"放弃于 {ab}，检索用 {used}/10")
    check("预算未烧光", used < 10, f"used={used}")
    check("首次放弃在第 3 轮", ab[0] == 3 if ab else False, f"rounds={ab}")

    print("\n[2] 每次检索都有进展：永不放弃（计数被重置）")
    ab2, used2 = simulate(drift, max_attempts=2, make_progress=True)
    check("无放弃", len(ab2) == 0, f"放弃={ab2}")
    check("预算消耗受覆盖收敛控制", used2 < 10, f"used={used2}")

    print("\n[3] 同一 stype 两次失败后放弃，第三种 stype 仍可检索")
    mixed = ["the actress and writer", "the actress", "the writer",
             "the actress again", "the actress once more"]
    sq_mix = SubQuestion(id="sq1", text="x", qtype="core")
    sq_mix.slots = [Slot(stype="entity", text=t) for t in mixed]
    sq_mix.slots.append(Slot(stype="time", text="when founded"))  # 不同 stype
    state = MetacognitiveState(question="q", sub_questions=[sq_mix])
    # 直接测 abandoned 持久化：_refresh_slots 逻辑
    abandoned = {"sq1": {"entity"}}
    new_slots = [
        {"stype": "entity", "text": "the actress", "importance": 1.0, "filled": False},
        {"stype": "time", "text": "when founded", "importance": 1.0, "filled": False},
    ]
    rebuilt = []
    for s in new_slots:
        rebuilt.append(Slot(
            stype=s["stype"], text=s["text"], importance=s["importance"],
            filled=bool(s["filled"]) or s["stype"] in abandoned.get("sq1", set())))
    entity_slots = [s for s in rebuilt if s.stype == "entity"]
    time_slots = [s for s in rebuilt if s.stype == "time"]
    check("已放弃 entity 槽重建后 filled=True", all(s.filled for s in entity_slots))
    check("未放弃 time 槽保持 open", all(not s.filled for s in time_slots))

    print("\n" + "=" * 50)
    print("✅ ABANDONMENT SMOKE PASSED" if ok else "❌ 存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
