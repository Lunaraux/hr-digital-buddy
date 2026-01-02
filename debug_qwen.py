# debug_qwen.py
import sys
from backend.adapters.qwen_llm import QwenLLM

# 1. 查看模块对象
module = sys.modules['backend.adapters.qwen_llm']
print("Module file path:", module.__file__)
print("Module source lines around __init__:")

# 2. 直接读取文件内容，打印第 5～15 行
with open(module.__file__, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(5, min(15, len(lines))):
        print(f"{i+1:2}: {lines[i].rstrip()}")