"""基线提示词（全部真实调用；JSON 输出约定同 EASE prompts.py）。"""

COT_FINAL = """你是问答助手。请基于常识与内在知识直接回答，简短给出答案。
输出 JSON：{{"answer":"简短答案"}}
问题：{question}"""

RAG_ANSWER = """你是问答助手。基于以下检索上下文回答，简短给出答案；证据不足则如实说明。
输出 JSON：{{"answer":"简短答案"}}
检索上下文：
{context}
问题：{question}"""

IRCOT_STEP = """你是 IRCoT 检索推理器。交替推理与检索：
- 若已知线索足以给出答案：{{"thought":"...","query":""}}（query 为空表示停止）
- 否则给出下一步检索查询：{{"thought":"...","query":"检索查询"}}
当前已检索内容：
{context}
问题：{question}"""

REACT_STEP = """你是 ReAct 智能体。交替思考与行动：
- 行动 search：{{"thought":"...","action":"search","query":"检索查询"}}
- 已知答案：{{"thought":"...","action":"answer","answer":"简短答案"}}
已检索内容：
{context}
问题：{question}"""
