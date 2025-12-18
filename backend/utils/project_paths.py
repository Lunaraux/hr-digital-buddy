# backend/utils/project_paths.py
from pathlib import Path

def get_project_root() -> Path:
    """
    获取项目根目录（hr-digital-buddy/）
    无论从哪个目录启动脚本，都能正确返回根路径。
    """
    # 当前文件路径: .../backend/utils/project_paths.py
    # 上两级: .../backend/
    # 再上一级: .../hr-digital-buddy/ ← 项目根目录
    return Path(__file__).parent.parent.parent.resolve()

# 预计算常用路径
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
HR_DOCS_DIR = DATA_DIR / "hr_docs"

# 模型路径
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDING_MODEL_DIR = MODELS_DIR / "embedding" / "all-MiniLM-L6-v2"
LLM_MODEL_DIR = MODELS_DIR / "llm"  
# 指向 LLM 的最终子目录
QWEN2_MODEL_PATH = LLM_MODEL_DIR / "qwen" / "Qwen2-1.5B-Instruct"