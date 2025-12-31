from fastapi import APIRouter, Depends
from pydantic import BaseModel
from backend.core.config import settings
from backend.adapters.chroma_vector_store import ChromaVectorStore
from backend.adapters.qwen_llm import QwenLLM
from backend.services.rag_service import RAGService
import torch

router = APIRouter()

#  定义请求模型
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

#  安全的依赖工厂
def get_rag_service():
    vector_store = ChromaVectorStore()
    
    llm_device = "cuda" if (settings.use_gpu and torch.cuda.is_available()) else "cpu"
    
    llm = QwenLLM(
        model_path=settings.llm_model_path,
        device=llm_device
    )
    return RAGService(vector_store=vector_store, llm=llm)

#  类型安全的路由
@router.post("/ask")
def ask_question(
    request: AskRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    return rag_service.ask(request.question, request.top_k)