"""配置加载：读取 config/default.yaml，并把 ${ENV_VAR} 替换为 .env / 环境变量中的真实值。

所有密钥只存在于 .env，不硬编码到代码或配置仓库文件。
"""
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_env():
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def load_config(path=None):
    load_env()
    cfg_path = path or os.path.join(PROJECT_ROOT, "config", "default.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config 文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        text = f.read()

    def _sub(m):
        key = m.group(1)
        val = os.environ.get(key, "")
        if not val:
            raise ValueError(
                f"config 引用了环境变量 {key} 但 .env 中为空/未设置（请检查项目根 .env）"
            )
        return val

    text = _ENV_RE.sub(_sub, text)
    cfg = yaml.safe_load(text)
    return cfg
