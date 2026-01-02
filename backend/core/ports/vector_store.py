# backend/core/ports/vector_store.py
from typing import Protocol, List
from dataclasses import dataclass

@dataclass
class RetrievedDocument:
    content: str
    metadata: dict
    score: float  # 建议统一为“相似度”，越大越好

class VectorStoreProtocol(Protocol):
    def similarity_search(self, query: str, k: int) -> List[RetrievedDocument]:
        ...