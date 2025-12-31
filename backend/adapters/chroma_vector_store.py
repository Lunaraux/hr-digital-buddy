# adapters/chroma_vector_store.py
from backend.core.ports.vector_store import VectorStoreProtocol, RetrievedDocument
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.core.config import settings
import torch


class ChromaVectorStore(VectorStoreProtocol):
    def __init__(self):
        if settings.use_gpu and torch.cuda.is_available():
            actual_device = "cuda"
        else:
            actual_device = "cpu"

        self.embedding = HuggingFaceEmbeddings(
            model_name=str(settings.embedding_model_path),
            model_kwargs={"device": actual_device}
        )
        self.db = Chroma(
            persist_directory=str(settings.chroma_path),
            embedding_function=self.embedding
        )

    def add_texts(self, texts: list[str], metadatas: list[dict]) -> None:
        self.db.add_texts(texts, metadatas)

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
        # Chroma 自动持久化，无需手动调用
        pass