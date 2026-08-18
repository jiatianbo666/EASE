"""预算规划与账本（P2/P12/P14）。

BudgetPlanner 持有：总检索次数、总 token 预算、最终化保留比例、每子问题分配。
- 分配：有技能先验用其 budget_ratio；否则按子问题权重贪心（Predictive Scheduling 式）
- 账本 ledger()：注入每一步 LLM 调用（BATS 式，P2）
- 最终化保留：FINALIZATION_RESERVE 比例 token 搜索永不可触碰（P12）
- 重分配：早停释放的检索次数转给未覆盖子问题（Think Just Enough）
"""
from dataclasses import dataclass, field


@dataclass
class BudgetPlanner:
    total_searches: int = 10
    total_tokens: int = 200_000
    finalization_reserve: float = 0.18
    conflict_allowance: int = 2

    searches_used: int = 0
    tokens_used: int = 0
    allocations: dict = field(default_factory=dict)   # subq_id -> searches
    forced_wrap_up: bool = False

    # ---------- 分配 ----------
    def plan(self, sub_questions, skill_prior=None):
        """为子问题分配检索次数。优先技能先验预算比例。"""
        ratios = None
        if (skill_prior is not None
                and getattr(skill_prior, "budget_ratio", None)
                and len(skill_prior.budget_ratio) == len(sub_questions)):
            ratios = skill_prior.budget_ratio
        else:
            w = [max(s.weight, 0.1) for s in sub_questions]
            tot = sum(w) or 1.0
            ratios = [x / tot for x in w]
        alloc = {}
        used = 0
        for sq, r in zip(sub_questions, ratios):
            n = round(self.total_searches * r)
            if sq.qtype == "core" and n == 0:
                n = 1
            alloc[sq.id] = n
            used += n
        # 溢出修正：从 follow_up 匀回 core（保证核心覆盖）
        for sq in reversed(sub_questions):
            while used > self.total_searches and alloc.get(sq.id, 0) > 0 and sq.qtype != "core":
                alloc[sq.id] -= 1
                used -= 1
        self.allocations = alloc
        return alloc

    # ---------- 账本 ----------
    def ledger(self):
        return (f"searches used {self.searches_used}/{self.total_searches} · "
                f"tokens {self.tokens_used}/{self.total_tokens} · "
                f"remaining {self.remaining_searches()} searches · "
                f"final-reserve {self.reserve_tokens()}")

    def remaining_searches(self):
        return self.total_searches - self.searches_used

    def remaining_tokens(self):
        return self.total_tokens - self.tokens_used

    def reserve_tokens(self):
        return int(self.total_tokens * self.finalization_reserve)

    # ---------- 消耗 ----------
    def consume_search(self):
        self.searches_used += 1

    def consume_tokens(self, n):
        self.tokens_used += n

    def budget_exhausted(self):
        return self.searches_used >= self.total_searches

    def can_use_tokens(self, n):
        """搜索阶段可用 token（扣除最终化保留）。"""
        return self.tokens_used + n <= self.total_tokens - self.reserve_tokens()

    # ---------- 重分配 ----------
    def reallocate(self, freed_subq_ids, target_subq_id):
        """把早停子问题剩余配额转给目标子问题。返回新 allocations。"""
        freed = sum(self.allocations.get(i, 0) for i in freed_subq_ids)
        for i in freed_subq_ids:
            self.allocations[i] = 0
        if freed > 0 and target_subq_id in self.allocations:
            self.allocations[target_subq_id] += freed
        return self.allocations
