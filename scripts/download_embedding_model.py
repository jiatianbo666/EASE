"""从 ModelScope 下载 bge-small-en-v1.5 到本地 data/models/。

真实下载（不走 huggingface.co，已被墙）。下载后校验文件大小。
模型命名空间：AI-ModelScope/bge-small-en-v1.5（ModelScope 官方，HTTP 200 已验证）
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from ease.utils.config import PROJECT_ROOT

NAMESPACE = "AI-ModelScope/bge-small-en-v1.5"
REVISION = "master"
DEST_DIR = PROJECT_ROOT / "data" / "models" / "bge-small-en-v1.5"

# sentence-transformers 加载所需的文件（跳过 pytorch_model.bin，用 safetensors；跳过 README）
FILES = [
    ("config.json", 743),
    ("config_sentence_transformers.json", 124),
    ("modules.json", 349),
    ("sentence_bert_config.json", 52),
    ("1_Pooling/config.json", None),
    ("tokenizer.json", 711396),
    ("tokenizer_config.json", 366),
    ("special_tokens_map.json", 125),
    ("vocab.txt", 231508),
    ("model.safetensors", 133466304),
]


def resolve_url(path):
    return f"https://www.modelscope.cn/models/{NAMESPACE}/resolve/{REVISION}/{path}"


def download(path, expected_size):
    dest = DEST_DIR / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        actual = dest.stat().st_size
        if expected_size is None or actual == expected_size:
            print(f"  [skip] {path} (已存在, {actual} bytes)")
            return True
        print(f"  [re-download] {path}: 大小不符 {actual} != {expected_size}")
    url = resolve_url(path)
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    actual = dest.stat().st_size
    ok = (expected_size is None) or (actual == expected_size)
    print(f"  {'✅' if ok else '⚠️'} {path}: {actual} bytes (期望 {expected_size})")
    return ok


def main():
    print(f"下载 {NAMESPACE} @ {REVISION} -> {DEST_DIR}")
    all_ok = True
    for path, size in FILES:
        try:
            if not download(path, size):
                all_ok = False
        except Exception as e:
            all_ok = False
            print(f"  ❌ {path}: {e}")
    print("=" * 50)
    if all_ok:
        print("✅ embedding 模型下载完成，可直接用 SentenceTransformer(local) 加载")
    else:
        print("⚠️ 部分文件下载失败，请重试或检查网络")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
