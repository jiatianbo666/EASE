"""Phase2 技能库改造验证 —— 全部离线（真实嵌入 + fake-LLM，零 API 成本）。

覆盖（对应 plan 验证清单）：
  T1 门控：retrieve(..., task_type=...) 硬门 + min_sim 阈值
  T2 derive 元数据：decomp 带 qtype、query 带 stype；from_dict 旧数据归一
  T3 注入裁剪：query_meta 按 stype 过滤表达式 + 无匹配不注入
  T4 成败对比：成功+失败 episode → LLM DO/DON'T 规则，save 后回读存在
  T5 真实诊断：失败 run 的 downvote reason 是诊断而非模板
  T6 保活：同名字覆写后 task_rules 不清
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ease.utils.config import load_config
from ease.embeddings.embedder import Embedder
from ease.experience.skill import Skill
from ease.experience.skill_store import SkillStore
from ease.experience.extractor import (
    RunSummary, SubQSummary, derive_skill, build_updates,
    build_contrast_rules, _abstract_query, episode_to_dict, episode_from_dict,
)
from ease.experience.episode_pool import EpisodePool

TMP_DB = "data/skills/verify_phase2.db"


class FakeLLM:
    """零成本：只响应 CONTRAST_RULES（含"成败对比"）与 DECOMPOSE 提示。"""
    def chat_json(self, role, messages, schema_hint=None, budget_ledger=None, **kw):
        p = messages[0]["content"]
        if "成败对比" in p:
            return {"rules": [
                "DO: 对比型题对每个候选实体单独检索其属性页，再比对属性。",
                "DON'T: 证据充足后继续追加检索，会烧预算且引入无关实体。",
                "DO: 属性槽查询必须携带已解析实体全名，避免歧义命中。",
            ]}
        if "参考历史经验" in p:
            return {"sub_questions": [{"text": "core fact", "qtype": "core"}]}
        return {}


def make_store():
    if os.path.exists(TMP_DB):
        os.remove(TMP_DB)
    cfg = load_config()
    cfg["experience"]["skills_db"] = TMP_DB
    return SkillStore(cfg["experience"], embedder=Embedder())


def T1_gate():
    store = make_store()
    s1 = Skill(name="skill-single", task_type="single",
               description="Use when answering single multi-hop questions: one core chain of facts.")
    s2 = Skill(name="skill-comparison", task_type="comparison",
               description="Use when answering comparison questions: compare two entities on a shared attribute.")
    assert store.add_skill(s1) == "inserted" and store.add_skill(s2) == "inserted"
    # 门语义：查询描述语义上偏向 comparison（是 comparison 技能的近邻），
    # 但当前题 task_type=single → 结构上只可能命中 single 技能，绝无 comparison。
    hit = store.retrieve("comparison question comparing two film directors",
                         top_k=2, task_type="single", min_sim=0.0)
    assert hit and [h.name for h in hit] == ["skill-single"], \
        f"异型技能泄漏: {[h.name for h in hit]}"
    # 对称：当前题 task_type=comparison → 只命中 comparison
    hit2 = store.retrieve("single question about one entity",
                          top_k=2, task_type="comparison", min_sim=0.0)
    assert hit2 and [h.name for h in hit2] == ["skill-comparison"], \
        f"comparison 门未生效: {[h.name for h in hit2]}"
    # min_sim 阈值兜底：同类型内相似度不足也拦截
    high = store.retrieve("single multi-hop question in film: some actor",
                          top_k=1, task_type="single", min_sim=0.99)
    assert high == [], "min_sim 高阈值未生效"
    # 旧调用（无 task_type）仍工作
    old = store.retrieve("single multi-hop question in film: some actor", top_k=1)
    assert old and old[0].name == "skill-single"
    print("  T1 门控 ✅ (异型拦截 / 对称门 / 阈值兜底 / 旧签名兼容)")
    store.close()
    os.remove(TMP_DB)


def T2_metadata_and_compat():
    run = RunSummary(
        qid="q1", task_type="comparison", question="Were X and Y same nationality?", success=True,
        sub_questions=[
            SubQSummary("locate X nationality", "core", 2, True),
            SubQSummary("locate Y nationality", "core", 2, True),
            SubQSummary("compare", "follow_up", 1, True),
        ],
        high_value_queries=["X nationality", "Y birthplace"],
        high_value_query_slots=["relation", "relation"],
    )
    sk = derive_skill(run, "skill-comparison")
    assert all(isinstance(d, dict) and "qtype" in d and "text" in d for d in sk.decomposition_template), \
        "decomp 条目未带 qtype"
    assert sk.query_meta and sk.query_meta[0]["stype"] == "relation", "query_meta 未带 stype"
    assert [d["qtype"] for d in sk.decomposition_template] == ["core", "core", "follow_up"]
    print("  T2 derive 元数据 ✅ (qtype/stype 入库)")

    # 旧数据向后兼容：纯字符串 decomp、无 query_meta/task_rules
    old_dict = {
        "name": "skill-single", "task_type": "single",
        "description": "old",
        "decomposition_template": ["find the entity", "answer"],
        "query_templates": ["some query"],
        "budget_ratio": [0.5, 0.5],
    }
    s = Skill.from_dict(old_dict)
    assert s.decomposition_template == [{"qtype": "core", "text": "find the entity"},
                                        {"qtype": "core", "text": "answer"}], s.decomposition_template
    assert s.query_meta == [] and s.task_rules == []
    print("  T2 from_dict 旧数据归一 ✅")


def T7_abstract_query():
    # 断点1：带训练实体名的具体查询 → 抽象成 <entityN> 骨架，异题注入不再泄漏
    query = ("Robin R. Bottin is known for his collaboration with an American director "
             "and producer who won an Academy Award, Golden Globe and BAFTA award for "
             "what movie?")
    anchors = ["Robin R. Bottin", "Danny Boyle", "American Beauty"]
    q2 = _abstract_query(query, anchors)
    assert "Robin R. Bottin" not in q2, f"训练实体未抽象: {q2}"
    assert "<entity" in q2, f"无占位符: {q2}"
    print(f"  T7 抽象化 ✅ ({q2[:80]}…)")

    # 单 token 实体（答案/值词）不抽象，保留查询可用性
    assert _abstract_query("dutch film 2006", ["Dutch"]) == "dutch film 2006"
    # 空锚点原样保留
    assert _abstract_query("who starred in Black Book", []) == "who starred in Black Book"
    print("  T7 抽象化边界 ✅ (单token/空锚点)")


def T3_injection_scoping():
    prior = Skill(query_meta=[{"stype": "time", "text": "<entity> founded year"},
                              {"stype": "entity", "text": "who starred in <film>"},
                              {"stype": "relation", "text": "<film> director"}])
    # 当前槽是 time → 只注入 time 模板
    slot_stype = "time"
    q_templates = [m["text"] for m in prior.query_meta if m.get("stype") == slot_stype][:3]
    assert q_templates == ["<entity> founded year"], q_templates
    # 当前槽是 numeric（无匹配）→ 不注入（宁缺勿滥）
    slot_stype = "numeric"
    q_templates = [m["text"] for m in prior.query_meta if m.get("stype") == slot_stype][:3]
    assert q_templates == [], "无匹配 stype 时应不注入"
    print("  T3 注入裁剪 ✅ (time 只注入 time 模板 / numeric 无匹配不注入)")


def T4_T5_T6():
    store = make_store()
    # 成功 episode
    succ = RunSummary(
        qid="s1", task_type="comparison", question="Were X and Y the same nationality?",
        success=True,
        sub_questions=[SubQSummary("locate X", "core", 2, True),
                       SubQSummary("locate Y", "core", 2, True),
                       SubQSummary("compare", "follow_up", 1, True)],
        high_value_queries=["X nationality", "Y birthplace"],
        high_value_query_slots=["relation", "relation"],
        final_coverage=1.0, actual_stop_round=3, total_cost=0.003,
    )
    # 失败 episode：槽放弃 + 覆盖卡住
    fail = RunSummary(
        qid="f1", task_type="comparison", question="Were A and B the same nationality?",
        success=False,
        sub_questions=[SubQSummary("locate A", "core", 3, False),
                       SubQSummary("locate B", "core", 3, False)],
        high_value_queries=[], low_value_queries=["A nationality", "B birthplace"],
        abandoned_slots=["sq2:entity"], final_coverage=0.5, actual_stop_round=6,
        total_cost=0.02, reused_skills=["skill-comparison"],
    )
    # T4 对比提炼（fake-LLM）→ 规则且入库回读
    rules_map = build_contrast_rules([succ, fail], FakeLLM())
    assert "comparison" in rules_map and rules_map["comparison"], rules_map
    sk = derive_skill(succ, "skill-comparison")
    sk.task_rules = rules_map["comparison"]
    assert store.add_skill(sk) == "inserted"
    reloaded = store.get("skill-comparison")
    assert reloaded and reloaded.task_rules == rules_map["comparison"], "task_rules 未入库/回读"
    print(f"  T4 对比提炼 ✅ ({len(reloaded.task_rules)} 条 DO/DON'T 入库回读)")

    # T5 真实诊断：失败 run 的 downvote reason 含具体槽/覆盖，不是模板
    ups = build_updates(fail)
    dvs = [u for u in ups if u.op == "downvote"]
    assert dvs, "失败 run 应有 downvote"
    assert "sq2:entity" in dvs[0].reason and dvs[0].reason != "reuse_failure: 复用技能但最终答案错误", dvs[0].reason
    print(f"  T5 真实诊断 ✅ ({dvs[0].reason})")

    # T6 保活：同名字 derive 覆写不清 task_rules
    sk2 = derive_skill(succ, "skill-comparison")   # 新技能（task_rules 空）
    assert store.add_skill(sk2) == "merged", "同描述应走 merge"
    assert store.get("skill-comparison").task_rules == rules_map["comparison"], "merge 清掉了 task_rules"
    print("  T6 保活 ✅ (merge/覆写后 task_rules 保留)")
    store.close()
    os.remove(TMP_DB)


def T8_episode_pool():
    """断点3：持久化 episode 池 + 周期增量提炼语义。"""
    import tempfile
    tmp = tempfile.mkdtemp()
    pool_path = os.path.join(tmp, "episodes.jsonl")

    def mk_run(qid, tt, success, nsub=2):
        return RunSummary(
            qid=qid, task_type=tt, question=f"{qid} question", success=success,
            sub_questions=[SubQSummary(f"sub{i}", "core", 2, success) for i in range(nsub)],
            high_value_queries=[f"{qid} query"], high_value_query_slots=["relation"],
            abandoned_slots=[] if success else ["sq1:entity"],
            final_coverage=1.0 if success else 0.4, actual_stop_round=3, total_cost=0.004,
            reused_skills=[f"skill-{tt}"],
        )
    ep_c_ok = mk_run("c1", "comparison", True)
    ep_c_fail = mk_run("c2", "comparison", False)
    ep_s_ok = mk_run("s1", "single", True)

    store = make_store()
    store.add_skill(Skill(name="skill-comparison", task_type="comparison", description="cmp skill"))
    store.add_skill(Skill(name="skill-single", task_type="single", description="single skill"))

    pool = EpisodePool(pool_path)
    assert pool.add([ep_c_ok, ep_c_fail, ep_s_ok]) == 3, "首次 add 应新增 3"
    assert pool.add([ep_c_ok]) == 0, "重复 qid 应去重"
    assert pool.count() == 3, "池内应有 3 条"

    # 序列化往返：dict → RunSummary 保真
    assert episode_from_dict(episode_to_dict(ep_c_ok)).qid == "c1"
    assert episode_from_dict(episode_to_dict(ep_c_fail)).abandoned_slots == ["sq1:entity"]

    # 首次提炼：comparison(成败都有→LLM) + single(只成功→确定性)，各写 task_rules
    s1 = pool.refresh_contrast(store, FakeLLM())
    assert s1["llm_calls"] == 1, f"只有 comparison 应有 LLM 调用: {s1}"
    assert s1["types"] == 2 and s1["rules"] >= 4, f"两类型都应有规则: {s1}"
    assert store.get("skill-comparison").task_rules, "comparison 规则未写入技能"
    assert store.get("skill-single").task_rules, "single 规则未写入技能"

    # 增量：无新 episode → 跳过（不重复烧 LLM）
    s2 = pool.refresh_contrast(store, FakeLLM())
    assert s2["types"] == 0 and s2["llm_calls"] == 0, f"无新增应跳过: {s2}"

    # 新 episode 到账 → 只重提炼该类型
    ep_c_fail2 = mk_run("c3", "comparison", False)
    assert pool.add([ep_c_fail2]) == 1
    s3 = pool.refresh_contrast(store, FakeLLM())
    assert s3["llm_calls"] == 1 and s3["types"] == 1, f"只有 comparison 应重提炼: {s3}"

    # 持久化：新实例从磁盘重建，计数/去重仍正确
    pool2 = EpisodePool(pool_path)
    assert pool2.count() == 4, f"磁盘重建应 4 条: {pool2.count()}"
    assert pool2.add([ep_c_ok]) == 0, "磁盘重建后仍去重"
    # meta 保留 → 新实例 refresh 应跳过已提炼类型
    s4 = pool2.refresh_contrast(store, FakeLLM())
    assert s4["llm_calls"] == 0, f"meta 应防止重复提炼: {s4}"
    store.close()
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print(f"  T8 episode 池 ✅ (去重/序列化往返/增量跳过/新批次重提炼/磁盘重建/meta 保活)")


def main():
    print("== Phase2 技能库改造验证 ==")
    T1_gate()
    T2_metadata_and_compat()
    T3_injection_scoping()
    T4_T5_T6()
    T7_abstract_query()
    T8_episode_pool()
    print("\n✅ ALL PHASE2 VERIFY TESTS PASSED (real embeddings · fake-LLM · zero API cost)")


if __name__ == "__main__":
    main()
