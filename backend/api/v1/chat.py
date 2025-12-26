# backend/api/v1/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.rag_service import RAGService

router = APIRouter()
_rag_service = RAGService()  # 单例，全局复用


class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_hr_question(request: QuestionRequest):
    if not request.question.strip():
        return {
            "answer": "问题不能为空",
            "sources": [],
            "audio_text": ""
        }
    return _rag_service.ask(request.question)