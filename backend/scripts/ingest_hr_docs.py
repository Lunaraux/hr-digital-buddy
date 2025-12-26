# backend/scripts/ingest_hr_docs.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.rag.vectorstore import VectorStoreManager


def load_hr_docs() -> List[Document]:
    docs: List[Document] = []
    hr_dir = "data/hr_docs"
    if not os.path.exists(hr_dir):
        print(f"目录不存在: {hr_dir}")
        return docs

    for file in os.listdir(hr_dir):
        if file.endswith(".txt"):
            path = os.path.join(hr_dir, file)
            try:
                loader = TextLoader(file_path=path, encoding="utf-8")
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
                print(f"加载: {file}")
            except Exception as e:
                print(f"加载 {file} 失败: {e}")
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

    vs = VectorStoreManager()
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]  # 包含 source 路径

    vs.ingest_texts(texts, metadatas)
    print("文档已成功加载到向量数据库！")

if __name__ == "__main__":
    main()