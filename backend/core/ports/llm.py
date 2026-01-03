# backend/core/ports/llm.py
from typing import Protocol

class LLMProtocol(Protocol):
    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        ...

    # 新增方法
    def generate_with_messages(
        self,
        user: str,
        system: str = "",
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        ...