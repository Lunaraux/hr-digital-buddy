# backend/services/rag_service.py
from typing import Dict, Any, List
from backend.core.ports.vector_store import VectorStoreProtocol, RetrievedDocument
from backend.core.ports.llm import LLMProtocol
import logging

logger = logging.getLogger(__name__)


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
        
        # Step 1: 检索相关文档
        try:
            docs: List[RetrievedDocument] = self.vector_store.similarity_search(question, k=top_k)
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return {
                "answer": "抱歉，检索文档时发生错误。",
                "sources": []
            }

        # Step 2: 找出第一条非空的有效文档
        selected_doc = None
        for d in docs:
            if d.content and d.content.strip():
                selected_doc = d
                break

        # Step 3: 构建上下文（只用第一条）
        if selected_doc:
            # 提取原始内容
            raw_content = selected_doc.content.strip()
            sources = [{"content": selected_doc.content, "metadata": selected_doc.metadata}]
            
            # 👇 关键：即使不截断，也确保上下文清晰（小模型能处理短文本）
            context = f"【HR政策原文】\n{raw_content}"
        else:
            context = "无相关资料。"
            sources = []

        # Step 4: 强化 system 指令 —— 精准区分模糊 vs 具体问题
        system_message = (
            "你是专业的人力资源助手，请严格按以下规则回答：\n"
            "1. 如果用户问题未说明具体工作年限（例如：‘年假多久？’、‘年假有几天？’），\n"
            "   请完整回答：‘年假天数根据工龄确定：入职满1年不满10年为5天，满10年不满20年为10天，满20年以上为15天。’\n"
            "2. 如果用户明确提到工作年限（例如：‘我工作3年’、‘入职8年’、‘干了15年’），\n"
            "   请根据政策匹配并仅输出对应天数（如‘5天’、‘10天’、‘15天’），不要解释。\n"
            "3. 禁止回答‘根据现有资料无法确定’，禁止随意猜测或只选最大值。\n"
            "4. 不得编造政策中没有的内容。"
        )
        user_message = f"参考资料：\n{context}\n\n问题：{question}"

        # Step 5: 调用 LLM
        try:
            answer = self.llm.generate_with_messages(
                system=system_message,
                user=user_message,
                max_tokens=128,
                temperature=0.0
            )
            # 清理可能的多余换行或前缀
            answer = answer.strip().split('\n')[0].strip('\"\'')
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "抱歉，生成答案时发生错误。"

        return {
            "answer": answer,
            "sources": sources
        }