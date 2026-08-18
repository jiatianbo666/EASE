"""全部提示词模板（task.md §8）。

约定：
- 所有小模型调用：thinking disabled + json_mode + 预算账本注入（由调用方处理）
- 模板用 str.format；JSON 字面量必须写成 {{ }} 转义
- 大模型只有 ANSWER_FINAL
"""

# ---- 阶段1 经验检索 ----
TASK_FEATURE = """你是 EASE 的任务特征提取器。分析这个问题，输出 JSON：
{{"task_type":"bridge|comparison|single|boolean|other","complexity":"low|medium|high","domain":"主题领域","entities":["关键实体"]}}
问题：{question}"""

# ---- 阶段2 分解 ----
DECOMPOSE = """你是 EASE 的子问题分解器。把问题拆成有向依赖的子问题。
规则：
- 只有确实需要多跳才拆 core；可直接从单一来源/常识回答的不拆。
- core=必须解决才能回答；background=补充背景；follow_up=整合比较。
输出 JSON：{{"sub_questions":[{{"text":"...","qtype":"core|background|follow_up"}}]}}
{few_shot}
问题：{question}"""

# ---- 缺口检测 ----
SLOT_GAP = """你是 EASE 的缺口检测器。给定子问题与当前证据，列出还缺什么信息。
每条槽：{{"stype":"entity|relation|numeric|time","text":"需要的信息","importance":0-1,"filled":bool}}
filled=true 必须已有证据支撑，否则 false。
输出 JSON：{{"slots":[...]}}
子问题：{subq}
现有证据：{evidence}"""

# ---- 查询生成 ----
QUERY_GEN = """你是 EASE 的精准查询生成器。针对一个信息缺口生成单一焦点搜索查询。
- 查询要具体（实体名完整），只针对这一个槽。
- 不要问废话，不要一次问多个问题。
{few_shot}
{resolved_entities}
输出 JSON：{{"query":"搜索查询","focus":"本查询要补的槽"}}
缺口槽：{slot_text}
子问题：{subq}"""

# 锚点块：querygen.py 在已解析实体非空时注入（提示词层让 LLM 自然锚定；
# 纯 python 后置回退兜底 LLM 漂移 —— 见 QueryGenerator.generate）
QUERY_GEN_ANCHORS = ("已解析实体：{entities}\n"
                     "- 若该缺口依赖某个已解析实体，必须把实体全名**逐字**嵌入查询，"
                     "禁止用代词/定冠词指代（the university / it / 该校）。")

# ---- 证据结构化抽取 ----
COMPRESS = """你是 EASE 的证据结构化器。把检索段落转成结构化事实。
- 只抽取与子问题相关的事实。
- 必须保留精确数值、日期、国籍等原子信息，不得改写。
- 每条：{{"ftype":"entity|relation|numeric|time|statement","subject":"主语","predicate":"谓语(可空)","value":"宾语/值","source_sent":"原始句子"}}
输出 JSON：{{"facts":[...]}}
子问题：{subq}
段落标题：{title}
段落：{text}"""

# ---- 覆盖度裁判 ----
COVERAGE_JUDGE = """你是 EASE 的覆盖度裁判。判断子问题是否已被现有证据块充分回答。
严格要求：
- 子问题所需的全部关键信息（数值、年份、名称、关系）必须**明确出现在证据文本中**。
- 缺失任何一个关键数字/年份/名称都算未覆盖（answered=false）。
- answered=true 必须有至少一个证据块直接给出答案，不能靠推断或常识。
输出 JSON：{{"answered":bool,"supporting_evidence":["证据块eid"]}}
子问题：{subq}
证据块：
{evidence_blocks}"""

# ---- 置信度探测 ----
CONFIDENCE = """你是 EASE 的置信度探测器。给定子问题与证据，给出提议答案与 P(true)（0-1）。
输出 JSON：{{"proposed_answer":"...","p_true":0.0-1.0}}
子问题：{subq}
证据：{evidence}"""

# ---- 停止安全监控 ----
STOP_SAFE_MONITOR = """你是 EASE 的停止安全监控。基于现有证据判断能否安全停止搜索。
- any_core_unaddressed: 是否有 core 子问题仍未得到证据回答
- any_conflict: 是否有未决的证据矛盾
输出 JSON：{{"any_core_unaddressed":bool,"any_conflict":bool}}
证据摘要：{evidence_summary}"""

# ---- 信息价值估计 ----
VOI_ESTIMATE = """你是 EASE 的信息价值估计器。估计获得某缺口信息后，改变最终答案的概率 p_change（0-1）。
输出 JSON：{{"p_change":0.0-1.0,"reason":"简述"}}
缺口：{slot_text}
当前状态：{state_summary}"""

# ---- 成败对比提炼规则（phase2 Fix 3：warm 预训练后逐 task_type 提炼）----
CONTRAST_RULES = """你是 EASE 的成败对比提炼器。同一任务类型下，给定成功轨迹与失败轨迹，提炼出区分两者的可执行规则。
要求：
- 每条以 DO 或 DON'T 开头，具体到可执行（查询怎么构造/子问题怎么拆/何时停止/槽怎么处理）。
- 3-5 条，每条不超过 120 字符。
- 规则必须能泛化到同类新题，不得包含任何具体题目的实体名。
输出 JSON：{{"rules":["..."]}}
任务类型：{task_type}
成功轨迹（{n_success} 条）：
{success_traces}
失败轨迹（{n_failure} 条）：
{failure_traces}"""

# ---- 大模型最终生成 ----
ANSWER_FINAL = """你是 EASE 的最终答案生成器。基于结构化证据生成带引用答案。
要求：
- answer 必须是最小化直接答案（单个实体/数值/年份/短语），与标准答案风格一致；不要解释性整句。
- 剥离修饰：若答案实体在证据中带年份括号或前缀（如 "The Changing Scottish Landscape (1991)"、"The 1991 publication …"），只输出核心实体本身，去掉括号内容与年份/修饰前缀。
- 答案优先取自证据事实。若证据缺少精确答案 token 但存在与该问题直接相关的实体/数值（如大学的建立年份、某人的全名），允许用背景知识**补全**（不得与证据矛盾），并在 unaddressed 注明是推断补全。
- 禁止用背景知识**替换**证据中已有的相关实体：证据已给出答案实体时，输出证据中的实体；不要换成知识联想里的其他实体（作者、年份、同领域人物）。
- 候选消歧：若证据中有多个直接回答问题的候选实体，用问题里的全部限定词（国籍/类型/年份/关系修饰语）逐个筛选，只输出满足全部限定词的那一个；仍分不出则取证据支持最多的一侧，另一侧写入 unaddressed。
- 类型对齐：答案的实体类型必须与问题问项一致。问施动者（by who / characterized by / wrote）时，对到结构化事实的 subject；不要把作者、年份等附属修饰当答案。
- 即使证据不完整（覆盖度偏低），也要基于现有证据给出最可能的答案；仅当证据与该问题完全无关（零相关事实）时才回答 Unknown；"证据不足" 不等于 "拒绝作答"，缺口写进 unaddressed。
- 每个关键事实用 [eid] 标注来源证据。
输出 JSON：{{"answer":"最短直接答案","cites":["eid"],"unaddressed":["..."]}}
问题：{question}
证据：
{evidence_context}
元认知摘要：
{state_summary}"""

# ---- 拒绝式回答重试（证据优先：部分证据也作答）----
ANSWER_RETRY = """你上次的回答是拒绝式的（"{prev}"）。请重新审视证据，给出基于证据的最可能答案。
- 只要证据中存在任何与该问题相关的事实，就从中选出最可能的答案（最小化直接答案，不加前缀修饰；答案实体若带年份括号/前缀，只输出实体本身）。
- 答案优先取自证据事实；证据缺少精确答案 token 但存在相关实体/数值时，允许用背景知识补全（不得与证据矛盾），但绝不能用知识替换证据中已有的相关实体。
- 多个候选实体都能回答时，用问题里的限定词筛选；多个事实互相矛盾时，取证据量更大的一侧，并在 unaddressed 披露另一侧。
- 仅当证据与该问题完全无关（零相关事实）时才保留 Unknown。
问题：{question}
证据：
{evidence_context}
元认知摘要：
{state_summary}"""

# ---- 多候选答案专注消歧（确定性单提交；answer.py 检测到列表时调用）----
ANSWER_COMMIT = """你是 EASE 的候选消歧器。你之前给出的答案是多个候选：
{candidates}
问题要求**恰好一个**答案。
消歧规则（按优先级，必须遵守）：
1. **限定词排除优先**：用问题里的全部限定词（国籍/类型/年份/关系修饰语等）逐个核对候选。
   国籍/类型/年份/关系不符的候选**直接淘汰**，即使它在证据中出现更多次或排在列表前面。
2. 剩余候选中，取证据支持最多的一侧。
3. 只输出一个实体/数值/年份/短语，不要列表、不要连接词、不要解释。
问题：{question}
证据：
{evidence_context}
元认知摘要：
{state_summary}"""
