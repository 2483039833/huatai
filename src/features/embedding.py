"""
文本向量化模块：BGE 模型加载 + embedding 生成 + 缓存
"""
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import settings


class EmbeddingEngine:
    """向量化引擎"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model_name
        self.device = settings.embedding_device
        self.batch_size = settings.embedding_batch_size
        self.max_seq_length = settings.embedding_max_seq_length
        self.dimension = settings.embedding_dimension
        self.cache_dir = Path(settings.data_embeddings_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None

    @property
    def model(self):
        """延迟加载模型（首次调用时才加载）"""
        if self._model is None:
            print(f"[Embedding] 加载模型: {self.model_name} on {self.device}")
            from sentence_transformers import SentenceTransformer
            t0 = time.time()

            # 1. 若传了本地路径则直接加载
            if Path(self.model_name).exists():
                self._model = SentenceTransformer(self.model_name, device=self.device)

            # 2. 否则优先使用本机 ModelScope 缓存（跳过不稳定的 HF 连接）
            elif self._local_modelscope_path() is not None:
                local_path = self._local_modelscope_path()
                print(f"[Embedding] 使用本机 ModelScope 缓存: {local_path}")
                self._model = SentenceTransformer(str(local_path), device=self.device)

            # 3. 国内网络优先从 ModelScope 下载，HF 仅作为最后兜底。
            else:
                try:
                    self._model = self._load_from_modelscope()
                except (OSError, ConnectionError, RuntimeError) as e:
                    print(f"[Embedding] ModelScope 加载失败({e}), 尝试 Hugging Face...")
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )

            self._model.max_seq_length = self.max_seq_length
            print(f"[Embedding] 模型加载完成 ({time.time() - t0:.1f}s)")
        return self._model

    def _local_modelscope_path(self) -> Optional[Path]:
        """查找本机 ModelScope 缓存中的模型路径（若存在）"""
        from pathlib import Path as P
        cache_root = P.home() / ".cache" / "modelscope"
        ms_model_id = self.model_name
        if ms_model_id.startswith("BAAI/"):
            ms_model_id = "AI-ModelScope/" + ms_model_id[len("BAAI/"):]

        # 新结构: models/<ID--with-dashes>/snapshots/master
        cand = cache_root / "models" / ms_model_id.replace("/", "--") / "snapshots" / "master"
        if cand.exists():
            return cand
        # 旧结构: hub/<ID_underscored>
        cand = cache_root / "hub" / ms_model_id.replace("/", "_")
        if cand.exists():
            return cand
        return None

    def _load_from_modelscope(self):
        """从 ModelScope 魔搭镜像加载 BGE 模型"""
        from pathlib import Path
        cache_root = Path.home() / ".cache" / "modelscope"

        # ModelScope 上的模型 ID（HF ID → ModelScope ID 映射）
        ms_model_id = self.model_name
        if ms_model_id.startswith("BAAI/"):
            ms_model_id = "AI-ModelScope/" + ms_model_id[len("BAAI/"):]

        # 检查本机是否已有缓存（ModelScope 缓存结构: models/<ID--with-dashes>/snapshots/master）
        model_dir = cache_root / "models" / ms_model_id.replace("/", "--") / "snapshots" / "master"
        if not model_dir.exists():
            # 兼容旧结构 hub/<ID_underscored>
            model_dir = cache_root / "hub" / ms_model_id.replace("/", "_")
        if model_dir.exists():
            print(f"[Embedding] 使用本机 ModelScope 缓存: {model_dir}")
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(str(model_dir), device=self.device)

        # 尝试用 modelscope 库下载
        try:
            from modelscope import snapshot_download
            local_dir = snapshot_download(
                ms_model_id,
                cache_dir=str(cache_root / "hub"),
            )
            print(f"[Embedding] ModelScope 下载完成: {local_dir}")
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(local_dir, device=self.device)
        except ImportError:
            raise RuntimeError(
                "ModelScope 未安装，请先执行: pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
            )
        except Exception as e:
            raise RuntimeError(
                f"无法加载模型 {self.model_name}，请检查网络或手动下载模型到本地后设置 model_name 为本地路径。原始错误: {e}"
            )

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        将文本列表转为向量

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            shape (n, dimension) 的 numpy 数组
        """
        clean_texts = [str(t) if t else "" for t in texts]
        embeddings = self.model.encode(
            clean_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # 归一化，余弦相似度可用内积
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_with_cache(
        self,
        texts: list[str],
        cache_key: str,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """
        带缓存的向量化：如果缓存文件存在则直接加载

        Args:
            texts: 文本列表
            cache_key: 缓存文件名（不含扩展名）
            force_recompute: 是否强制重新计算

        Returns:
            embedding numpy array
        """
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if not force_recompute and cache_path.exists():
            print(f"[Embedding] 从缓存加载: {cache_path}")
            return np.load(str(cache_path))

        embeddings = self.encode(texts)

        np.save(str(cache_path), embeddings)
        print(f"[Embedding] 缓存已保存: {cache_path}")
        return embeddings

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        对 DataFrame 某一列做向量化

        Args:
            df: 输入 DataFrame
            text_column: 文本列名
            cache_key: 可选缓存 key

        Returns:
            embedding numpy array
        """
        texts = df[text_column].fillna("").tolist()
        if cache_key:
            return self.encode_with_cache(texts, cache_key)
        return self.encode(texts)
