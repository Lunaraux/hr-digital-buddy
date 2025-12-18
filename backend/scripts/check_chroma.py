# backend/scripts/check_chroma.py
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入类（不是函数！）
from backend.rag.vectorstore import VectorStoreManager

# 创建实例（会自动加载 Chroma 和 Embedding 模型）
vs_manager = VectorStoreManager()

# 直接访问底层 Chroma 对象来获取原始文档
collection = vs_manager.vectorstore._collection
results = collection.get(include=["documents"])
docs = results["documents"]

print(f"向量库中共有 {len(docs)} 段文本：\n")
for i, doc in enumerate(docs):
    print(f"[{i+1}] {doc[:200]}...\n")