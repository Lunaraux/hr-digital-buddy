# adapters/chroma_vector_store.py
from backend.core.ports.vector_store import VectorStoreProtocol, RetrievedDocument
from langchain_chroma import Chroma
from backend.core.config import settings
from backend.core.embedding_manager import get_embedding_model


class ChromaVectorStore(VectorStoreProtocol):
    def __init__(self):
        self.embedding = get_embedding_model()
        self.db = Chroma(
            persist_directory=str(settings.chroma_path),
            embedding_function=self.embedding
        )
    def add_texts(self, texts: list[str], metadatas: list[dict]) -> None:
        self.db.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 3) -> list[RetrievedDocument]:
        results = self.db.similarity_search_with_score(query, k=k)
        return [
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score)
            )
            for doc, score in results
        ]

    def persist(self) -> None:
        pass  # Chroma 自动持久化