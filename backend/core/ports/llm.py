from typing import Protocol

class LLMProtocol(Protocol):
    """LLM 接口协议，定义语言模型的标准行为"""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """生成回答

        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本回复
        """
        ...
