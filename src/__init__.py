"""项目核心实现包。

`src/` 目录承担了这条 Vision RAG 主链路的大部分底层实现：
1. `config.py` 负责集中管理环境变量和运行参数。
2. `doc_processor.py` 负责把 PDF、图片、文本统一转换成页面图像。
3. `vector_store.py` 负责用 ColPali + MUVERA + Qdrant 建索引与检索。
4. `llm_generator.py` 负责把证据页和问题组织成多模态请求，并调用大模型生成答案。

阅读建议：
先看 `config.py` 了解系统有哪些基础配置，再看 `doc_processor.py`
理解输入如何标准化，随后进入 `vector_store.py` 和 `llm_generator.py`
把“检索”和“生成”两段主逻辑串起来。
"""
