# backend/build_kb.py
import pickle
import os
from pathlib import Path

import jieba
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from backend.core.config import settings

# 加载自定义词典（模块级执行，OK）
jieba.load_userdict(str(settings.base_dir / "data" / "hr_dict.txt"))

def main():
    # 1. 读取文档
    docs, metadatas = [], []
    for file in settings.hr_docs_path.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            docs.append(content)
            metadatas.append({"source": str(file.relative_to(settings.base_dir))})

    if not docs:
        print("⚠️ 未找到任何 HR 文档，请检查 data/hr_docs/ 目录")
        return

    print(f"📄 读取到 {len(docs)} 份文档")

    # 2. 确定 embedding 模型来源
    if (settings.embedding_model_path / "modules.json").exists():
        model_source = str(settings.embedding_model_path)
        print(f"📁 使用本地 embedding 模型: {model_source}")
    else:
        model_source = "BAAI/bge-large-zh-v1.5"
        print(f"☁️ 本地模型未找到，使用在线模型: {model_source}")

    # 3. 创建 embedding function
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=model_source,
        device=settings.embedding_device
    )

    # 4. 初始化客户端
    client = PersistentClient(path=str(settings.chroma_path))

    # 5. 删除并重建集合（确保干净）
    print("🗑️  删除旧集合...")
    try:
        client.delete_collection("hr_policies")
    except Exception:
        pass  # 忽略“集合不存在”错误

    print("🆕 创建新集合...")
    collection = client.create_collection(
        name="hr_policies",
        embedding_function=embedding_fn
    )

    # 6. 添加文档
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(docs))]
    )

    # 7. 保存原始文档供 BM25 使用
    raw_docs_path = settings.base_dir / "data" / "raw_docs.pkl"
    with open(raw_docs_path, "wb") as f:
        pickle.dump({"documents": docs, "metadatas": metadatas}, f)

    print(f"✅ 知识库构建完成！")
    print(f"   向量库存储于: {settings.chroma_path}")
    print(f"   原始文档缓存: {raw_docs_path}")

if __name__ == "__main__":
    main()