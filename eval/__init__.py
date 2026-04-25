"""VisionRAG 带标注检索质量评估包。

包含两个评估器：
  - retrieval_eval  检索层质量评估（Recall@k、MRR、NDCG@5）
  - answer_eval     回答有据可依评估（grounding + 人工核对答案）

使用前必须先准备带金标的查询文件，详见 eval/queries/ANNOTATION_GUIDE.md。
"""
