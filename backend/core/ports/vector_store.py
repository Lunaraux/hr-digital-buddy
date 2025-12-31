from typing import Protocol, List
from dataclasses import dataclass

@dataclass
class RetrievedDocument:
    content: str
    metadata: dict
    score: float

class VectorStoreProtocol(Protocol):
    def add_texts(self, texts: List[str], metadatas: List[dict]) -> None: ...
    def similarity_search(self, query: str, k: int = 3) -> List[RetrievedDocument]: ...
    def persist(self) -> None: ...