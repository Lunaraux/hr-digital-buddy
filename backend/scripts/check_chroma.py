# backend/scripts/check_chroma.py
from chromadb.api.models.Collection import Collection


import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入类
from backend.rag.vectorstore import VectorStoreManager


def main():
    try:
        # 创建实例（会自动加载 Chroma 和 Embedding 模型）
        vs_manager: VectorStoreManager = VectorStoreManager()

        # 直接访问底层 Chroma 集合
        collection: Collection = vs_manager.vectorstore._collection

        # 获取所有文档（包括内容、metadata、ids）
        results = collection.get(include=["documents", "metadatas", "embeddings"])

        docs = results["documents"]
        metadatas = results.get("metadatas", [{}] * len(docs))  # pyright: ignore[reportArgumentType]

        print(f"向量库中共有 {len(docs)} 段文本：\n")  # pyright: ignore[reportArgumentType]

        for i, (doc, meta) in enumerate(zip(docs, metadatas)):  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]
            source = meta.get("source", "未知来源")
            preview = doc[:200].replace("\n", " ").strip()
            print(f"[{i+1}] 来源: {source}")
            print(f"     内容: {preview}...\n")

    except Exception as e:
        print(f"检查向量库时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()