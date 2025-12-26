# backend/services/rag_service.py
from typing import List, Dict, Any
from langchain_core.documents import Document
from backend.rag.vectorstore import VectorStoreManager
from backend.models.llm import HRLLM

# Prompt 定义在 RAGService 所在模块
_RAG_PROMPT_TEMPLATE = """你是一名专业的人力资源助手，请根据以下公司政策内容回答问题。
要求：
- 如果下方“公司政策内容”为空、为“无”、或不包含与问题相关的信息，请直接回答：“抱歉，未找到相关信息。”
- 不要编造、推测或使用外部知识
- 用中文简洁、友好地回答

公司政策内容：
{context}

问题：{question}"""


class RAGService:
    def __init__(self):
        self._vs = VectorStoreManager()
        self._llm = HRLLM()

    def ask(self, question: str) -> Dict[str, Any]:
        docs: List[Document] = self._vs.retrieve(question, k=3)
        context = "\n".join(doc.page_content for doc in docs)
        sources = [
            {
                "source": self._extract_filename(doc.metadata.get("source", "")),
                "content": doc.page_content
            }
            for doc in docs
        ]

        if not context.strip():
            return {
                "answer": "抱歉，未找到相关信息。",
                "sources": [],
                "audio_text": "未找到相关信息"
            }

        # 构造 prompt
        prompt = _RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        answer_text = self._llm.generate(prompt)  # 传入完整 prompt

        return {
            "answer": answer_text,
            "sources": sources,
            "audio_text": answer_text
        }

    @staticmethod
    def _extract_filename(path: str) -> str:
        return path.split("/")[-1].split("\\")[-1] if path else "未知来源"