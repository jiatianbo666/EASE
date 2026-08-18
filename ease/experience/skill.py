"""经验层 Skill —— 检索面 + 载荷（progressive disclosure）。

检索面（被嵌入/检索的部分）：name, description, task_type, 统计
载荷（命中后才加载）：decomposition_template, budget_ratio, query_templates, stop_params
"""
from dataclasses import dataclass, field, fields, asdict


@dataclass
class Skill:
    name: str = ""
    description: str = ""            # ≤512 字符，含 "Use when..." 触发词
    task_type: str = ""              # bridge / comparison / ...
    # ---- 载荷 ----
    decomposition_template: list = field(default_factory=list)  # 子问题分解 few-shot（条目 {qtype,text}）
    budget_ratio: list = field(default_factory=list)           # 子问题预算分配比例（和=1）
    query_templates: list = field(default_factory=list)        # 高价值查询模板（旧格式，新格式用 query_meta）
    query_meta: list = field(default_factory=list)             # 带槽型标签的查询模板 [{"stype","text"}]，注入按 stype 裁剪
    task_rules: list = field(default_factory=list)             # task 级 DO/DON'T 规则（成败对比提炼）
    stop_params: dict = field(default_factory=dict)            # {max_rounds, delta_cov_eps, ...}
    # ---- 统计（ExpeL 生命周期）----
    success_rate: float = 0.0
    avg_cost: float = 0.0            # 美元
    usage_count: int = 0
    importance: int = 2              # 新=2；UPVOTE+1；DOWNVOTE-1；到 0 删除
    failure_lessons: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        valid = {f.name for f in fields(Skill)}
        s = Skill(**{k: v for k, v in d.items() if k in valid})
        # 向后兼容：旧 decomposition_template 是纯字符串 → 包成 {qtype,text}
        norm_decomp = []
        for item in s.decomposition_template:
            if isinstance(item, dict):
                text = str(item.get("text", ""))
                if text:
                    norm_decomp.append({"qtype": str(item.get("qtype", "core")), "text": text})
            elif item:
                norm_decomp.append({"qtype": "core", "text": str(item)})
        s.decomposition_template = norm_decomp
        # query_meta 归一为 {stype,text}；非法条目丢弃（宁缺勿滥，注入不碰）
        norm_meta = []
        for item in s.query_meta:
            if isinstance(item, dict) and item.get("text"):
                norm_meta.append({"stype": str(item.get("stype", "")), "text": str(item["text"])})
        s.query_meta = norm_meta
        return s

    def summary(self):
        """检索面（控制层注入用）：不含载荷。"""
        return {
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "success_rate": round(self.success_rate, 3),
            "avg_cost": round(self.avg_cost, 6),
            "usage_count": self.usage_count,
            "importance": self.importance,
        }
