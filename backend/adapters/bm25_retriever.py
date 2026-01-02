# backend/adapters/bm25_retriever.py
import pickle
import jieba
from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from backend.core.config import settings


class BM25Retriever:
    def __init__(self):
        raw_docs_path = settings.base_dir / "data" / "raw_docs.pkl"
        with open(raw_docs_path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

        # 分词构建索引
        tokenized_docs = [list(jieba.cut(doc)) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            (self.documents[i], self.metadatas[i], float(scores[i]))
            for i in top_indices
        ]