# backend/scripts/ingest_hr_docs.py
import sys
from pathlib import Path
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入你实际实现的向量存储适配器
from backend.adapters.chroma_vector_store import ChromaVectorStore


def load_hr_docs() -> List[Document]:
    docs: List[Document] = []
    hr_dir = PROJECT_ROOT / "data" / "hr_docs"
    if not hr_dir.exists():
        print(f"目录不存在: {hr_dir}")
        return docs

    for file in hr_dir.iterdir():
        if file.suffix == ".txt":
            try:
                loader = TextLoader(file_path=str(file), encoding="utf-8")
                loaded_docs = loader.load()
                # 确保 metadata 中有 source（LangChain 通常会自动加）
                docs.extend(loaded_docs)
                print(f"加载: {file.name}")
            except Exception as e:
                print(f"加载 {file.name} 失败: {e}")
    return docs


def main() -> None:
    docs = load_hr_docs()
    if not docs:
        print("data/hr_docs/ 目录下没有找到 .txt 文件！")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    chunks: List[Document] = text_splitter.split_documents(docs)

    print(f"原始文档数: {len(docs)}")
    print(f"切分为文本块: {len(chunks)}")

    # 直接使用 ChromaVectorStore
    vs = ChromaVectorStore()
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vs.add_texts(texts, metadatas)  # ← 使用 add_texts，不是 ingest_texts
    print("文档已成功加载到向量数据库！")


if __name__ == "__main__":
    main()