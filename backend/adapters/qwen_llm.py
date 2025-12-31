# backend/adapters/qwen_llm.py
from pathlib import Path
from typing import Union
from backend.core.ports.llm import LLMProtocol
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QwenLLM(LLMProtocol):
    def __init__(self, model_path: Union[str, Path], device: str = "cuda"):
        # 转为字符串，确保 transformers 兼容
        model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map=device
        )
        self.device = device

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        #  构造符合 Qwen 聊天模板的消息格式
        messages = [{"role": "user", "content": prompt}]
        
        #  必须通过 self.tokenizer 实例调用 apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        #  tokenize 输入
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096 - max_tokens
        ).to(self.device)

        #  生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        #  解码输出（仅新生成部分）
        input_length = model_inputs.input_ids.shape[1]
        response = self.tokenizer.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=True
        )[0]

        return response.strip()