from backend.core.ports.vector_store import VectorStoreProtocol, RetrievedDocument
from backend.core.ports.llm import LLMProtocol

class RAGService:
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        llm: LLMProtocol
    ):
        self.vector_store = vector_store
        self.llm = llm

    def ask(self, question: str, top_k: int = 3) -> dict:
        docs: list[RetrievedDocument] = self.vector_store.similarity_search(question, k=top_k)
        context = "\n\n".join([d.content for d in docs])
        prompt = f"""基于以下 HR 政策内容回答问题：
{context}

问题：{question}
回答："""
        answer = self.llm.generate(prompt)
        sources = [{"content": d.content, "metadata": d.metadata} for d in docs]
        return {"answer": answer, "sources": sources}