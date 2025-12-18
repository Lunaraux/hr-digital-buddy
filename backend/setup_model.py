from modelscope import snapshot_download
import os

# # 指定干净的下载路径
# target_dir = r"D:\MyProgram\hr-digital-buddy\models\embedding"

# print("开始从 ModelScope 下载 all-MiniLM-L6-v2...")
# local_model_path = snapshot_download(
#     'sentence-transformers/all-MiniLM-L6-v2',
#     cache_dir=target_dir
# )

# print(f" 模型已完整下载到: {local_model_path}")

# 确保目录存在
os.makedirs(r"D:\MyProgram\hr-digital-buddy\models\llm", exist_ok=True)

print("开始从 ModelScope 下载 Qwen2-1.5B-Instruct...")
local_model_path = snapshot_download(
    'qwen/Qwen2-1.5B-Instruct',  # ⭐ ModelScope 上的模型ID
    cache_dir=r"D:\MyProgram\hr-digital-buddy\models\llm"
)

print(f"✅ LLM 已完整下载到: {local_model_path}")