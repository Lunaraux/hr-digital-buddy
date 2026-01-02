# backend/core/embedding_manager.py
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from backend.core.config import settings

_embedding_instance = None

def get_embedding_model():
    global _embedding_instance
    if _embedding_instance is None:
        device = "cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu"
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=str(settings.embedding_model_path),
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_instance