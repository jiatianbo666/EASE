"""容错 JSON 解析（借鉴 GenericAgent llmcore.tryparse 的思路）。

模型输出的 JSON 常常带 ```json 代码块、前后说明文字、甚至截断。
本模块负责从中稳定提取最外层 JSON 对象/数组；解析失败返回 None（不抛异常）。
"""
import json
import re


def parse_json_tolerant(text):
    """从模型输出中提取并解析最外层 JSON 对象/数组。

    处理顺序：
      1. 去除 ```json ... ``` 代码块
      2. 尝试截取最外层 {..} 或 [..] 解析
      3. 直接整体解析
    全部失败返回 None。
    """
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        t = m.group(1).strip()

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        i = t.find(open_ch)
        j = t.rfind(close_ch)
        if i != -1 and j > i:
            cand = t[i:j + 1]
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None
