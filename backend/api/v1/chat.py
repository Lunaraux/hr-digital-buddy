from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.rag_service import ask_hr_question

router = APIRouter(prefix="/v1/chat", tags=["Chat"])

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
def ask(request: QuestionRequest):
    return {"answer": ask_hr_question(request.question)}