"""本地 bge-small-en-v1.5 嵌入器（真实模型，GPU，归一化向量）。"""
import os
import time

import numpy as np

from ..utils.config import PROJECT_ROOT


class Embedder:
    def __init__(self, model_dir=None, device="cuda", batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.model_dir = model_dir or str(PROJECT_ROOT / "data" / "models" / "bge-small-en-v1.5")
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"embedding 模型不存在：{self.model_dir}\n"
                "请先运行 scripts/download_embedding_model.py"
            )
        t0 = time.time()
        self.model = SentenceTransformer(self.model_dir, device=device)
        self.device = device
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.load_seconds = time.time() - t0

    def embed(self, texts, batch_size=None, progress=False):
        """输入 str 或 list[str]，返回 float32 (N, dim)，L2 归一化。"""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=progress,
        )
        return np.asarray(vecs, dtype=np.float32)
