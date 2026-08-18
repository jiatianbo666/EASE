"""M3 真实验证 —— SkillStore：真实嵌入检索 + ExpeL 投票生命周期 + 去重合并。

用全新临时 DB，插入两条手写真实技能，验证：
  1) 向量检索命中最相似技能
  2) UPVOTE/DOWNVOTE 到删除的完整生命周期
  3) 描述余弦去重 → merged
  4) 每次变更后 assert_integrity 一致
  5) extractor 从成功轨迹派生技能
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config
from ease.embeddings.embedder import Embedder
from ease.experience.skill import Skill
from ease.experience.skill_store import SkillStore
from ease.experience.extractor import RunSummary, SubQSummary, derive_skill


def main():
    cfg = load_config()
    # 临时 DB（覆盖上次验证残留）
    tmp_db = "data/skills/smoke_test.db"
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    cfg["experience"]["skills_db"] = tmp_db

    store = SkillStore(cfg["experience"], embedder=Embedder())

    # ---- 1. 插入两条真实技能 ----
    s1 = Skill(
        name="bridge-multihop",
        task_type="bridge",
        description=("Use when answering bridge multi-hop questions: connect two entities "
                     "through an intermediate fact. Decompose into locating entity A, locating "
                     "entity B, then finding the bridging relation. Allocate most budget to the "
                     "intermediate hop."),
        decomposition_template=["locate entity A", "locate entity B", "find bridging relation"],
        budget_ratio=[0.3, 0.3, 0.4],
        query_templates=["<EntityA> <relation> <EntityB>", "<EntityA> birth date"],
    )
    s2 = Skill(
        name="comparison-two-people",
        task_type="comparison",
        description=("Use when answering comparison questions: compare two entities on a "
                     "shared attribute (nationality, date, award). Retrieve a page per entity, "
                     "extract the attribute from each, then judge equality."),
        decomposition_template=["extract attribute of entity A", "extract attribute of entity B", "compare"],
        budget_ratio=[0.4, 0.4, 0.2],
    )
    r1 = store.add_skill(s1, verified=True)
    r2 = store.add_skill(s2, verified=True)
    print(f"插入技能: bridge={r1}, comparison={r2}")
    assert r1 == "inserted" and r2 == "inserted"

    # ---- 2. 检索最相似 ----
    print("\n[检索] 查询描述: comparison question comparing two film directors")
    hits = store.retrieve("comparison question comparing two film directors", top_k=2)
    for h in hits:
        print(f"  [{h.name}] task_type={h.task_type} desc={h.description[:60]}...")
    assert hits and hits[0].name == "comparison-two-people", "检索未命中最相似技能"

    # 未验证写入被拒
    assert store.add_skill(Skill(name="ghost", description="should not be admitted"), verified=False) == "skipped"

    # ---- 3. 去重合并 ----
    dup = Skill(name="comparison-copy", task_type="comparison",
                description=("Use when answering comparison questions: compare two entities on a "
                             "shared attribute (nationality, date, award). Retrieve a page per entity, "
                             "extract the attribute from each, then judge equality."))
    r3 = store.add_skill(dup, verified=True)
    print(f"\n[去重] 相似描述插入: {r3}")
    assert r3 == "merged", f"期望 merged，实际 {r3}"
    assert store.count() == 2, "去重后不应新增条目"

    # ---- 4. 投票生命周期 ----
    print("\n[生命周期] comparison-two-people: 初始 importance=2")
    s = store.get("comparison-two-people")
    assert s.importance == 2
    store.upvote("comparison-two-people")
    assert store.get("comparison-two-people").importance == 3
    print("  +UPVOTE → 3")
    r = store.downvote("comparison-two-people", reason="test lesson")
    assert r is True and store.get("comparison-two-people").importance == 2
    r = store.downvote("comparison-two-people", reason="still failing")
    assert r is True and store.get("comparison-two-people").importance == 1
    print("  -DOWNVOTE×2 → 1 (保留 failure_lessons)")
    assert store.get("comparison-two-people").failure_lessons == ["test lesson", "still failing"]
    r = store.downvote("comparison-two-people")
    assert r == "deleted" and store.get("comparison-two-people") is None
    print("  -DOWNVOTE×1 → 0 → 已删除")

    # ---- 5. extractor 从成功轨迹派生技能 ----
    run = RunSummary(
        qid="q1", task_type="bridge", question="Were X and Y of same nationality?", success=True,
        sub_questions=[
            SubQSummary("locate X", "core", 2, True),
            SubQSummary("locate Y", "core", 3, True),
            SubQSummary("compare", "follow_up", 1, True),
        ],
        high_value_queries=["X nationality", "Y birthplace"],
        optimal_stop_round=3, actual_stop_round=3, total_cost=0.001,
    )
    sk = derive_skill(run, "bridge-multihop")
    print(f"\n[extractor] 派生技能 {sk.name}: ratios={sk.budget_ratio}, lessons 无")
    assert abs(sum(sk.budget_ratio) - 1.0) < 1e-6

    # 失败 run → 对复用技能 downvote
    run_fail = RunSummary(qid="q2", task_type="bridge", question="...", success=False,
                          sub_questions=[], reused_skills=["bridge-multihop"])
    from ease.experience.extractor import build_updates, apply_updates
    ups = build_updates(run_fail)
    print("  失败 run 生成更新:", [(u.op, getattr(u.target, "name", u.target)) for u in ups])
    assert any(u.op == "downvote" for u in ups)

    print("\n" + "=" * 50)
    print(f"最终 integrity: {store.assert_integrity()}，技能数: {store.count()}")
    assert store.assert_integrity()
    store.close()
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    print("✅ ALL SKILLSTORE SMOKE TESTS PASSED (real embeddings · real SQLite · real lifecycle)")


if __name__ == "__main__":
    main()
