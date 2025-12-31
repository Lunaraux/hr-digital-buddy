# backend/core/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    base_dir: Path = Path(__file__).parent.parent.parent
    embedding_model_path: Path = base_dir / "models" / "embedding" / "all-MiniLM-L6-v2"
    llm_model_path: Path = base_dir / "models" / "llm" / "qwen" / "Qwen2-1.5B-Instruct"
    chroma_path: Path = base_dir / "data" / "chroma"
    hr_docs_path: Path = base_dir / "data" / "hr_docs"

    use_gpu: bool = True
    embedding_device: str = "cuda" if use_gpu else "cpu"
    llm_device: str = "cuda" if use_gpu else "cpu"

    default_top_k: int = 3
    max_answer_length: int = 512

    class Config:
        env_file = ".env"

settings = Settings()