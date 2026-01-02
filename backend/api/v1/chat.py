# backend/api/v1/chat.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from backend.core.config import settings
# from backend.adapters.chroma_vector_store import ChromaVectorStore  # 可注释掉
from backend.adapters.hybrid_store import HybridVectorStore            # 新增导入
from backend.adapters.qwen_llm import QwenLLM
from backend.services.rag_service import RAGService
import torch

router = APIRouter()

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

def get_rag_service():
    # vector_store = ChromaVectorStore()      # 纯向量（旧）
    vector_store = HybridVectorStore()        # 混合检索（新）
    
    llm_device = "cuda" if (settings.use_gpu and torch.cuda.is_available()) else "cpu"
    
    llm = QwenLLM(
        model_path=settings.llm_model_path,
        device=llm_device
    )
    return RAGService(vector_store=vector_store, llm=llm)

@router.post("/ask")
def ask_question(request: AskRequest):
    vector_store = HybridVectorStore()
    llm_device = "cuda" if (settings.use_gpu and torch.cuda.is_available()) else "cpu"
    llm = QwenLLM(model_path=settings.llm_model_path, device=llm_device)
    rag = RAGService(vector_store=vector_store, llm=llm)
    return rag.ask(request.question, request.top_k)