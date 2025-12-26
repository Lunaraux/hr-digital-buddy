# backend/rag/vectorstore.py
from pathlib import Path
from typing import Any, List, Dict
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.utils.project_paths import CHROMA_DIR, EMBEDDING_MODEL_DIR


# 模块级缓存：确保 embedding 模型只加载一次
_embedding_instance = None


def _get_embedding_function():
    """单例方式获取 embedding function,避免重复加载模型"""
    global _embedding_instance
    if _embedding_instance is None:
        model_path = str(EMBEDDING_MODEL_DIR)
        if not (EMBEDDING_MODEL_DIR / "config.json").exists():
            raise FileNotFoundError(
                f"Embedding 模型未找到: {model_path}\n"
                "请先将 all-MiniLM-L6-v2 模型放入 models/embedding/all-MiniLM-L6-v2/\n"
                "或运行 download_embedding.py 自动下载。"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding 模型加载中... (设备: {device})")

        _embedding_instance = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding 模型加载完成")
    return _embedding_instance


class VectorStoreManager:
    """管理 Chroma 向量数据库的初始化、写入与检索"""

    def __init__(self, persist_dir: str | None = None) -> None:
        if persist_dir is None:
            persist_dir = str(CHROMA_DIR)
        self.persist_dir = persist_dir
        
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        print(f"向量库存储路径: {self.persist_dir}")

        self.embedding = _get_embedding_function()
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )

    def ingest_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] | None = None) -> None:
        """
        注入文本到向量库。
        :param texts: 文本列表
        :param metadatas: 可选，每个文本的元数据（如 {"source": "leave_policy.txt"}）
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)  # 默认空 metadata，完全兼容旧调用
        documents = [
            Document(page_content=t, metadata=m)
            for t, m in zip(texts, metadatas)
        ]
        self.vectorstore.add_documents(documents)
        print(f"已存入 {len(documents)} 段文本到 {self.persist_dir}")

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """根据查询检索最相关的文档"""
        return self.vectorstore.similarity_search(query, k=k)