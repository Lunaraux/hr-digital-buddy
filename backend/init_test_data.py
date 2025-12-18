# backend/test_vectorstore.py
import sys
import os

# 确保能导入 backend（当从项目根目录运行时）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.project_paths import CHROMA_DIR
from backend.rag.vectorstore import VectorStoreManager

if __name__ == "__main__":
    print(f" 项目根目录: {os.getcwd()}")
    print(f" 向量库存储路径: {CHROMA_DIR}")

    # 初始化向量库
    vs = VectorStoreManager(persist_dir=str(CHROMA_DIR))

    # 注入测试数据
    test_texts = [
        "员工每年享有15天带薪年假,入职满一年后生效。",
        "病假需提供医院证明,每月最多3天带薪病假。",
        "加班需提前申请,工作日加班按1.5倍工资计算。"
    ]
    vs.ingest_texts(test_texts)
    print(" 测试数据注入完成！")

    # 立即检索验证
    query = "年假政策是什么？"
    print(f"\n 测试检索: '{query}'")
    results = vs.retrieve(query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content}")

    print("\n 测试成功！向量库已持久化，可被其他模块加载。")