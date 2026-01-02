# backend/adapters/reranker.py
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from backend.core.config import settings


class BGEReranker:
    def __init__(self, model_name_or_path: str = "BAAI/bge-reranker-base"):
        self.device = "cuda" if (settings.use_gpu and torch.cuda.is_available()) else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int = 3,
        batch_size: int = 32
    ) -> List[Tuple[str, float]]:
        if not docs:
            return []
        
        all_pairs = [[query, doc] for doc in docs]
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(all_pairs), batch_size):
                batch = all_pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                scores = self.model(**inputs).logits.view(-1).float().cpu()
                all_scores.extend(scores.tolist())

        scored = [(doc, score) for doc, score in zip(docs, all_scores)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]