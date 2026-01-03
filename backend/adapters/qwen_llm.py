# backend/adapters/qwen_llm.py
from pathlib import Path
from typing import Union
from backend.core.ports.llm import LLMProtocol
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

logger = logging.getLogger(__name__)


class QwenLLM(LLMProtocol):
    def __init__(self, model_path: Union[str, Path], device: str = "cuda"):
        model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device
        )
        self.device = device

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """兼容旧接口"""
        return self.generate_with_messages(user=prompt, max_tokens=max_tokens, temperature=0.0)

    def generate_with_messages(
        self,
        user: str,
        system: str = "",
        max_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """
        推荐使用此方法。
        - system: 系统角色指令（如“你是HR助手...”）
        - user: 用户输入（含问题和上下文）
        - temperature=0.0 → 确定性输出，适合小模型 RAG
        """
        messages = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=0.9 if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = model_inputs.input_ids.shape[1]
        response = self.tokenizer.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=True
        )[0]

        return response.strip()