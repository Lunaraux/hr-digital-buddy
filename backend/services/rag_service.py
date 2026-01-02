from typing import Dict, Any, List
from backend.core.ports.vector_store import VectorStoreProtocol, RetrievedDocument
from backend.core.ports.llm import LLMProtocol
import logging

logger = logging.getLogger(__name__)
MAX_CONTEXT_LENGTH = 8000

class RAGService:
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        llm: LLMProtocol
    ):
        self.vector_store = vector_store
        self.llm = llm

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        logger.info(f"Processing question: {question[:100]}...")
        
        docs: List[RetrievedDocument] = self.vector_store.similarity_search(question, k=top_k)
        
        # 处理无结果情况
        if not docs:
            return {
                "answer": "根据现有资料无法确定",
                "sources": []
            }

        # 构建上下文（防超长）
        context_parts = []
        total_len = 0
        for d in docs:
            if total_len + len(d.content) > MAX_CONTEXT_LENGTH:
                break
            context_parts.append(d.content)
            total_len += len(d.content)
        context = "\n\n".join(context_parts)

        prompt = f"""【角色】你是一个 HR 政策问答助手，必须严格遵守以下规则：

【规则】
1. 仅使用下方【政策原文】中明确写出的内容回答。
2. 若原文未提及、未定义或无法直接得出答案，必须回答：“根据现有资料无法确定”。
3. 禁止：
   - 使用外部知识、常识或经验推断
   - 混淆不同条款（如将年假规则用于试用期）
   - 使用“通常”、“一般”、“可能”等模糊词
   - 总结、改写、扩展原文
4. 回答必须简洁，不超过两句话。

【政策原文】
{context}

【问题】
{question}

【回答】（严格按规则）：
"""

        answer = self.llm.generate(prompt, max_tokens=512)
        sources = [{"content": d.content, "metadata": d.metadata} for d in docs]
        
        logger.info(f"Generated answer: {answer[:100]}...")
        return {"answer": answer, "sources": sources}