"""EASE — Experience-aware, Adaptive, Search with Efficient budgeting."""
__version__ = "0.1.0"

import os

# 本环境 transformers 4.57 导入时会拉取 TensorFlow，而 TF 安装损坏（DLL 加载失败）。
# EASE 完全不需要 TF；必须在任何 transformers 导入前禁用。
# 实测 USE_TF=0 可正常加载 bge-small-en-v1.5（见 M2 验证记录）。
os.environ.setdefault("USE_TF", "0")
