# backend/rag/vectorstore.py
import os
from pathlib import Path
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.utils.project_paths import CHROMA_DIR, EMBEDDING_MODEL_DIR

from backend.config import HF_CACHE_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR


class VectorStoreManager:
    """管理 Chroma 向量数据库的初始化、写入与检索"""
    
    persist_dir: str
    embedding: HuggingFaceEmbeddings
    vectorstore: Chroma

    def __init__(self, persist_dir: str | None = None) -> None:
        if persist_dir is None:
            persist_dir = str(CHROMA_DIR)
        self.persist_dir = persist_dir
        
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        print(f"向量库存储路径: {self.persist_dir}")

        model_path = str(EMBEDDING_MODEL_DIR)
        if not (EMBEDDING_MODEL_DIR / "config.json").exists():
            raise FileNotFoundError(
                f"Embedding 模型未找到: {model_path}\n"  # pyright: ignore[reportImplicitStringConcatenation]
                "请先将 all-MiniLM-L6-v2 模型放入 models/embedding/all-MiniLM-L6-v2/\n"
                "或运行 download_embedding.py 自动下载。"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding 模型加载中... (设备: {device})")

        self.embedding = HuggingFaceEmbeddings(
            model_name=model_path,  # ← 本地路径
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding 模型加载完成")

        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )

    def ingest_texts(self, texts: list[str]) -> None:
        documents = [Document(page_content=t) for t in texts]
        self.vectorstore.add_documents(documents)
        print(f"已存入 {len(texts)} 段文本到 {self.persist_dir}")

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)