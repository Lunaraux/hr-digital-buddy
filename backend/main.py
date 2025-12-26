# backend/main.py
from fastapi import FastAPI
import torch

print("CUDA available:", torch.cuda.is_available())

app = FastAPI(title="HR Digital Buddy API")

from backend.api.v1.chat import router as chat_router

app.include_router(chat_router, prefix="/v1/chat")

@app.get("/health")
def health():
    return {"status": "ok"}