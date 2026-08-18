"""Router —— 动作选择（P1 无缺口不检索 / P10 低不确定度跳过检索 / P12 强制 wrap-up）。

决定 RETRIEVE / REASON / GENERATE / STOP，并给 RETRIEVE 绑定明确缺口。
纯逻辑状态机（决策基于元认知状态 + 预算），不直接调用 LLM。
"""
from dataclasses import dataclass


@dataclass
class Action:
    kind: str            # RETRIEVE | REASON | GENERATE | STOP
    gap: tuple = None    # (subq, slot) —— RETRIEVE 的绑定缺口
    reason: str = ""


class Router:
    def __init__(self, config):
        self.cov_stop = config.get("coverage_stop_threshold", 0.8)
        self.conf_stop = config.get("conf_stop_threshold", 0.75)

    def decide(self, state, budget, calib_confidence=None, gaps=None):
        """基于元认知状态与预算选择动作。

        优先级：
          1. 预算耗尽 / wrapping_up  → GENERATE（P12 强制 wrap-up）
          2. core 冲突未决           → RETRIEVE（第三方来源，消耗冲突配额）
          3. 存在 core 缺口          → RETRIEVE（绑定最高优先缺口）
          4. 覆盖度+置信度达标        → GENERATE
          5. 非 core 缺口且预算充足   → RETRIEVE（宽松）
          6. 兜底                    → GENERATE
        """
        if budget.budget_exhausted() or state.status == "wrapping_up":
            return Action("GENERATE", reason="budget exhausted / wrapping up")

        if state.has_unresolved_core_conflict() and budget.conflict_allowance > 0:
            return Action("RETRIEVE", reason="core conflict needs third source")

        if gaps is None:
            gaps = state.detect_gaps()
        core_gaps = state.core_gaps(gaps)

        if core_gaps and budget.remaining_searches() > 0:
            sq, slot = core_gaps[0]
            return Action("RETRIEVE", gap=(sq, slot),
                          reason=f"core gap [{slot.stype}] {slot.text[:60]}")

        if state.core_coverage() >= self.cov_stop and not core_gaps:
            conf = calib_confidence if calib_confidence is not None else self._avg_confidence(state)
            if conf >= self.conf_stop:
                return Action("GENERATE", reason=f"covered {state.core_coverage():.2f} + conf {conf:.2f}")

        if gaps and budget.remaining_searches() >= 2:
            sq, slot = gaps[0]
            return Action("RETRIEVE", gap=(sq, slot), reason="non-core gap (spare budget)")

        return Action("GENERATE", reason="fallback (no high-value gap)")

    @staticmethod
    def _avg_confidence(state):
        confs = [s.confidence for s in state.sub_questions]
        return (sum(confs) / len(confs)) if confs else 0.0
