# backend/adapters/hybrid_store.py
from typing import List
from backend.core.ports.vector_store import VectorStoreProtocol, RetrievedDocument
from chromadb import PersistentClient
from backend.adapters.bm25_retriever import BM25Retriever
from backend.core.config import settings


class HybridVectorStore(VectorStoreProtocol):  
    def __init__(self):
        self.client = PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_collection("hr_policies")
        self.bm25_retriever = BM25Retriever()

    def similarity_search(self, query: str, k: int = 3) -> List[RetrievedDocument]:  
        top_k = k
        alpha = 0.7  # 可考虑从 config 注入

        # 向量检索（Chroma 原生 API）
        vec_res = self.collection.query(query_texts=[query], n_results=top_k * 2)
        vec_docs = vec_res["documents"][0]
        vec_metas = vec_res["metadatas"][0]
        vec_dists = vec_res["distances"][0]

        # 转为相似度（余弦距离 → 相似度）
        vec_scores = [1.0 / (1.0 + d) for d in vec_dists]

        # BM25 检索
        bm25_results = self.bm25_retriever.retrieve(query, top_k * 2)
        bm25_dict = {meta["source"]: (doc, meta, score) for doc, meta, score in bm25_results}

        # 合并结果（去重 + 补全）
        merged = {}
        for doc, meta, score in zip(vec_docs, vec_metas, vec_scores):
            source = meta["source"]
            merged[source] = {
                "content": doc,
                "meta": meta,
                "vec_score": score,
                "bm25_score": bm25_dict.get(source, (None, None, 0.0))[2]
            }

        for source, (doc, meta, score) in bm25_dict.items():
            if source not in merged:
                merged[source] = {
                    "content": doc,
                    "meta": meta,
                    "vec_score": 0.0,
                    "bm25_score": score
                }

        # 归一化 + 加权融合
        vec_vals = [r["vec_score"] for r in merged.values()]
        bm25_vals = [r["bm25_score"] for r in merged.values()]
        max_v = max(vec_vals) if any(vec_vals) else 1.0
        max_b = max(bm25_vals) if any(bm25_vals) else 1.0

        results = []
        for r in merged.values():
            final_score = alpha * (r["vec_score"] / max_v) + (1 - alpha) * (r["bm25_score"] / max_b)
            results.append(
                RetrievedDocument(
                    content=r["content"],
                    metadata=r["meta"],          
                    score=final_score
                )
            )

        # 按分数降序，取 top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]