# backend/models/llm.py（完整修正版）
import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from backend.utils.project_paths import QWEN2_MODEL_PATH
# 使用项目本地模型，不再需要外部HF缓存


class HRLLM:
    _instance: "HRLLM | None" = None
    _initialized: bool = False

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel

    def __new__(cls) -> "HRLLM":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if HRLLM._initialized:
            return
        HRLLM._initialized = True

        print(" 正在加载 Qwen2-1.5B-Instruct 模型...")
        
        model_path = str(QWEN2_MODEL_PATH)
        if not (QWEN2_MODEL_PATH / "config.json").exists():
            raise FileNotFoundError(
                f"LLM 模型未找到: {model_path}\n"
                "请先将 Qwen2-1.5B-Instruct 模型放入 models/llm/qwen/Qwen2-1.5B-Instruct/\n"
                "或运行 setup_model.py 自动下载。"
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" 使用设备: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)  # pyright: ignore[reportArgumentType]

        self.model.eval()
        print(" Qwen2-1.5B 模型加载完成")

    def generate(self, prompt: str) -> str:
        """接收完整 prompt，不关心业务逻辑"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)  # pyright: ignore[reportArgumentType]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()