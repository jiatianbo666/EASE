"""M4 真实验证 —— 控制层：元认知覆盖度/缺口、预算分配/账本、路由、停止。

用合成 MetacognitiveState 驱动状态机（judge 用确定性 stub，LLM 判定在 agent 层接入），
验证核心决策逻辑：
  1) update_coverage → 覆盖度变化、状态转换
  2) detect_gaps → 权重×重要性排序
  3) BudgetPlanner：技能先验比例 / 兜底权重分配 / 最终化保留 / 账本串
  4) Router：core 缺口→RETRIEVE 绑定 / 预算耗尽→GENERATE / 覆盖达标→GENERATE
  5) StopChecker：边际增益早停 + safe-stop 拦截 / 放行
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.control.metacognitive import MetacognitiveState, SubQuestion, Slot
from ease.control.budget import BudgetPlanner
from ease.control.router import Router, Action
from ease.control.stop import StopChecker


class Ev:
    def __init__(self, eid, text):
        self.eid, self.raw_text = eid, text


def stub_judge(sq_id_map, answered_ids):
    """返回一个判定"只有指定子问题被覆盖"的 judge。"""
    def judge(subq_text, ev_texts):
        sq_id = sq_id_map.get(subq_text, "")
        return {"answered": sq_id in answered_ids,
                "supporting_evidence": [sq_id]}
    return judge


def main():
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        mark = "✅" if cond else "❌"
        print(f"  {mark} {name}{' | ' + detail if detail else ''}")
        if not cond:
            ok = False

    # ---------- 构造状态：core + background + follow_up ----------
    sq_core = SubQuestion(id="sq1", text="locate entity A", qtype="core",
                          slots=[Slot("entity", "A 的国籍")])
    sq_bg = SubQuestion(id="sq2", text="locate entity B", qtype="background",
                        slots=[Slot("entity", "B 的国籍")])
    sq_fu = SubQuestion(id="sq3", text="compare nationality", qtype="follow_up",
                        slots=[Slot("relation", "A 与 B 是否同国籍")])
    st = MetacognitiveState(question="Were A and B of the same nationality?",
                            sub_questions=[sq_core, sq_bg, sq_fu])

    print("[1] 覆盖度更新")
    sq_id_map = {s.text: s.id for s in st.sub_questions}
    st.update_coverage([Ev("ev1", "A 是美国人")], coverage_judge=stub_judge(sq_id_map, {"sq1"}))
    check("core 覆盖 → covered", sq_core.status == "covered", f"cov={st.core_coverage():.2f}")
    check("core_coverage 只算 core", abs(st.core_coverage() - 1.0) < 1e-9)
    st.update_coverage([], coverage_judge=stub_judge(sq_id_map, {"sq2"}))
    check("bg 覆盖", sq_bg.status == "covered")
    check("total_coverage 加权", abs(st.total_coverage() - 1.5/1.8) < 1e-9,
          f"= {st.total_coverage():.3f}（期望 1.5/1.8=0.833）")

    print("\n[2] 缺口检测（core 缺口优先级最高）")
    gaps = st.detect_gaps()
    check("检测到 follow_up 缺口", len(gaps) == 1 and gaps[0][0].id == "sq3",
          f"gaps={[(g[0].id, g[1].text) for g in gaps]}")
    # 让 core 变回 open 再测优先级
    sq_core.status, sq_core.coverage = "open", 0.0
    st.sub_questions[0] = sq_core
    gaps2 = st.detect_gaps()
    check("core 缺口排第一", gaps2 and gaps2[0][0].id == "sq1")

    print("\n[3] 预算分配与账本")
    bp = BudgetPlanner(total_searches=10, total_tokens=100_000, finalization_reserve=0.18)
    bp.plan(st.sub_questions, skill_prior=None)
    check("兜底分配存在", bp.allocations.get("sq1", 0) >= 1, str(bp.allocations))
    ledger = bp.ledger()
    check("账本串含预算/剩余", "searches used 0/10" in ledger and "final-reserve 18000" in ledger, ledger)
    bp.consume_search(); bp.consume_search()
    check("消耗后剩余", bp.remaining_searches() == 8)
    check("can_use_tokens 扣除保留", bp.can_use_tokens(10_000) and not bp.can_use_tokens(90_000))

    print("\n[4] 路由决策")
    router = Router({"coverage_stop_threshold": 0.8, "conf_stop_threshold": 0.75})
    a1 = router.decide(st, bp, gaps=st.detect_gaps())
    check("有 core 缺口 → RETRIEVE 绑定", a1.kind == "RETRIEVE" and a1.gap is not None,
          f"{a1.kind}: {a1.reason[:50]}")
    # 预算耗尽 → GENERATE
    bp2 = BudgetPlanner(total_searches=10, finalization_reserve=0.18)
    bp2.searches_used = 10
    a2 = router.decide(st, bp2)
    check("预算耗尽 → GENERATE", a2.kind == "GENERATE", a2.reason)
    # 覆盖达标 + 置信达标 → GENERATE
    st3 = MetacognitiveState(question="q", sub_questions=[
        SubQuestion("sq1", "a", "core", coverage=1.0, status="covered"),
        SubQuestion("sq2", "b", "background", coverage=1.0, status="covered"),
    ])
    a3 = router.decide(st3, BudgetPlanner(total_searches=10))
    check("覆盖+置信达标 → GENERATE", a3.kind == "GENERATE", a3.reason)
    # core 冲突 → RETRIEVE
    st4 = MetacognitiveState(question="q", sub_questions=[
        SubQuestion("sq1", "a", "core", coverage=1.0, status="covered")])
    st4.add_conflict("sq1", [["ev1"], ["ev2"]])
    a4 = router.decide(st4, BudgetPlanner(total_searches=10))
    check("core 冲突未决 → RETRIEVE 第三方", a4.kind == "RETRIEVE", a4.reason)

    print("\n[5] 停止检查（复合 + safe-stop）")
    sc = StopChecker(eps=0.02, window=2)
    sc.observe_coverage(1.0); sc.observe_coverage(1.0); sc.observe_coverage(0.4); sc.observe_coverage(0.4)
    st5 = MetacognitiveState(question="q", sub_questions=[
        SubQuestion("sq1", "a", "core", coverage=1.0, status="covered"),
        SubQuestion("sq2", "b", "background", coverage=0.0),
    ])
    stop1, r1 = sc.should_stop(st5, BudgetPlanner(total_searches=10),
                               safe_monitor=lambda s: {"any_core_unaddressed": False, "any_conflict": False})
    check("边际增益低 + safe 放行 → 停止", stop1, r1)
    stop2, r2 = sc.should_stop(st5, BudgetPlanner(total_searches=10),
                               safe_monitor=lambda s: {"any_core_unaddressed": True, "any_conflict": False})
    check("safe-stop 拦截（core 未答）→ 继续", not stop2, r2)
    # 覆盖达标 + 高置信 → 停
    sc2 = StopChecker(eps=0.02, window=2)
    sc2.observe_coverage(0.0); sc2.observe_coverage(0.6); sc2.observe_coverage(1.0)
    st6 = MetacognitiveState(question="q", sub_questions=[
        SubQuestion("sq1", "a", "core", coverage=1.0, status="covered"),
        SubQuestion("sq2", "b", "core", coverage=1.0, status="covered")])
    stop3, r3 = sc2.should_stop(st6, BudgetPlanner(total_searches=10), calib_confidence=0.9)
    check("core 全覆盖+置信高 → 停止", stop3, r3)

    print("\n" + "=" * 50)
    print("✅ ALL CONTROL SMOKE TESTS PASSED" if ok else "❌ 存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
