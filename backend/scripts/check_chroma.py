# backend/scripts/check_chroma.py
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, cast

# 添加项目根目录到路径（兼容直接运行）
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.adapters.chroma_vector_store import ChromaVectorStore


def main() -> None:
    try:
        vs_manager = ChromaVectorStore()
        collection = vs_manager.db._collection

        results = collection.get(include=["documents", "metadatas"])

        docs: List[str] = results.get("documents") or []
        # 使用 cast 明确告诉类型检查器：我们接受这个转换
        metadatas: List[Optional[Dict[str, Any]]] = cast(
            List[Optional[Dict[str, Any]]],
            results.get("metadatas") or [None] * len(docs)
        )

        print(f"向量库中共有 {len(docs)} 段文本：\n")

        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
            source = meta.get("source", "未知来源") if meta is not None else "未知来源"
            preview = doc[:200].replace("\n", " ").strip()
            print(f"[{i+1}] 来源: {source}")
            print(f"     内容: {preview}...\n")

    except Exception as e:
        print(f"检查向量库时出错: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()