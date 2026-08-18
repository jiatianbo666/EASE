"""StopChecker —— 复合停止规则 + safe-stop 监控（P8/P12；task.md §7.6 §9.6）。

STOP if ANY：
  (a) 覆盖度滚动边际增益 < eps 连续 window 轮
  (b) core 覆盖度 > 阈值 且 无 core 缺口 且 校准置信 > 阈值
  (c) 搜索预算耗尽（强制 wrap-up）
候选早停后 → safe-stop 监控复核（1 次小模型调用，注入实现）：
    仍有 core 子问题未答或证据矛盾 → 继续搜索。
"""
from dataclasses import dataclass, field


@dataclass
class StopChecker:
    eps: float = 0.02
    window: int = 2
    cov_stop: float = 0.8
    conf_stop: float = 0.75
    _hist: list = field(default_factory=list)   # 每轮 total_coverage

    def observe_coverage(self, total_cov):
        self._hist.append(total_cov)
        if len(self._hist) > 20:
            self._hist.pop(0)

    def rolling_gain(self):
        """近 window 轮的覆盖度边际增益（首段返回大值避免误停）。"""
        if len(self._hist) < 2:
            return 1.0
        recent = self._hist[-(self.window + 1):]
        return recent[-1] - recent[0]

    def should_stop(self, state, budget, safe_monitor=None, calib_confidence=None):
        """返回 (stop: bool, reason: str)。"""
        if state.status != "running":
            return True, f"state={state.status}"
        if budget.budget_exhausted():
            return True, "budget exhausted (wrap-up)"

        candidate, reason = False, ""
        core_gaps = state.core_gaps(state.detect_gaps())
        gain = self.rolling_gain()
        # 边际增益早停：仅当无未覆盖的 core 缺口才允许（确定性防线，
        # 不依赖小模型 safe-monitor —— 防止 core 缺口仍开就误停）
        if gain < self.eps and not core_gaps:
            candidate, reason = True, f"marginal gain {gain:.3f} < eps {self.eps} (no core gaps)"
        elif state.core_coverage() >= self.cov_stop:
            conf = calib_confidence if calib_confidence is not None else 0.0
            if not core_gaps and conf >= self.conf_stop:
                candidate, reason = True, f"core covered {state.core_coverage():.2f} + conf {conf:.2f}"

        if not candidate:
            return False, ""

        # safe-stop 监控（P8）：确认无 core 子问题未答/证据矛盾才真正停
        if safe_monitor is not None:
            verdict = safe_monitor(state)
            if verdict.get("any_core_unaddressed") or verdict.get("any_conflict"):
                return False, f"safe-stop: {verdict}"
        return True, reason
