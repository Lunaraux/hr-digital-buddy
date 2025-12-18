from backend.rag.vectorstore import VectorStoreManager
from backend.models.llm import HRLLM

_vs = None
_llm = None

def _get_vs():
    global _vs
    if _vs is None:
        _vs = VectorStoreManager()
    return _vs

def _get_llm():
    global _llm
    if _llm is None:
        _llm = HRLLM()
    return _llm

def ask_hr_question(question: str) -> str:
    docs = _get_vs().retrieve(question, k=3)
    context = "\n".join([d.page_content for d in docs])
    if not context.strip():
        return "抱歉，未找到相关信息。"
    return _get_llm().generate(context, question)