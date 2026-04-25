# 金标数据标注指南

本文档说明如何为 VisionRAG 检索评估准备 **带 gold-page 标注的查询文件**。
评估器会直接对比系统检索到的页面与你标注的 gold page，计算 Recall、MRR、NDCG。

---

## 1. 查询文件格式

文件：`eval/queries/gold_queries.jsonl`，每行一条 JSON，格式如下：

```json
{
  "query_id":        "q001",
  "query_text":      "这份文档描述的核心方法是什么？",
  "document_name":   "AttentionIsAllYouNeed.pdf",
  "gold_page_numbers": [2, 3],
  "primary_answer":  "提出了 Transformer 架构，完全基于注意力机制，取消了 RNN 和 CNN。",
  "query_type":      "factual",
  "difficulty":      "easy"
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `query_id` | 是 | 唯一字符串，建议 `q001`、`q002` 格式 |
| `query_text` | 是 | 真实用户会提问的自然语言问题 |
| `document_name` | 是 | **与系统中实际存储的文件名完全一致**（含扩展名）；查看方式见第 3 节 |
| `gold_page_numbers` | 是 | 含有答案的页码列表（**1-indexed**，即第 1 页为 1）；1 到 3 页 |
| `primary_answer` | 是 | 参考答案，1 到 3 句话，后续用于人工核对 |
| `query_type` | 是 | 见下方类型表 |
| `difficulty` | 是 | `easy` / `medium` / `hard` |

### query_type 分类

| 类型 | 定义 | 示例 |
|------|------|------|
| `factual` | 单页就能回答的事实型问题 | "这个模型使用了什么优化器？" |
| `process` | 涉及步骤或流程 | "训练流程的具体步骤是什么？" |
| `numeric` | 答案涉及具体数值、表格 | "在 WMT 任务上 BLEU 得分是多少？" |
| `limitation` | 局限性、注意事项 | "文中提到的主要局限是什么？" |
| `synthesis` | 跨页综合信息 | "方法与实验结果综合来看说明了什么？" |

---

## 2. 标注步骤（每份文档）

1. **确认文件名**：把 PDF 上传到 VisionRAG，在文件列表里找到它显示的文件名，**原样复制**到 `document_name` 字段。
2. **按渲染页序标注**：系统把 PDF 第 1 页渲染为 `page_1.png`，第 N 页为 `page_N.png`，与 PDF 自带页码一致。
3. **草拟问题**：先通读文档，然后写出 3 到 5 个真实用户可能提问的问题，覆盖上面 5 种类型。不要写出文档里没有明确答案的问题。
4. **标 gold page**：对每个问题，找出 1 到 3 个"仅靠这几页就足以回答"的页码。页码从 1 开始，对应 PDF 的实际页序。
5. **写参考答案**：用 1 到 3 句简洁中文写下标准答案。回答要能在 gold page 的图像中找到支撑。
6. **标注跨页问题**：若答案必须依赖多页，`gold_page_numbers` 里填上所有必要页；`query_type` 选 `synthesis`，`difficulty` 选 `hard`。
7. **一致性复核**：完成一份文档的标注后，随机抽取 20% 的条目重新核对，重点检查：
   - 页码是否从 1 开始（而不是从 0）
   - `document_name` 是否与系统文件列表中一模一样
   - 问题是否存在歧义（可以同时指向多个不同的答案）

---

## 3. 如何查看系统存储的 document_name

方式一（推荐）：启动后端，访问 `GET /api/rag/files`，返回的 JSON 中每条记录的 `"name"` 字段即为 `document_name`。

方式二：查看 Qdrant 存储的 payload，运行：
```bash
python - <<'EOF'
from src.vector_store import VisionVectorStore
store = VisionVectorStore()
for doc in store.get_all_documents():
    print(doc['document_name'])
EOF
```

---

## 4. 最小可用数据规模

| 场景 | 文档数 | 总 query 数 | 说明 |
|------|--------|------------|------|
| 快速验证 | 3–4 份 | 15–20 条 | 适合先跑通流程 |
| 正式基线 | 6–8 份 | 24–30 条 | 每种 query_type 各有 5+ 条 |
| 可信对比实验 | 8–10 份 | 40+ 条 | 含至少 10 条 synthesis 类 |

文档建议：8 到 20 页的技术文档或论文，内含正文、表格、图示、流程说明，且有跨页信息，使检索任务有难度。

---

## 5. 常见错误

- **页码从 0 开始**：系统页码从 1 开始，标 `[0]` 永远不会命中。
- **document_name 大小写不一致**：系统按精确字符串比对，`Report.pdf` ≠ `report.pdf`。
- **gold page 太宽泛**：把整篇文档的 10 页都标为 gold 会使评估失去意义，每题最多标 3 页。
- **答案只靠记忆写**：参考答案要基于 gold page 的实际内容，否则后续人工核对没有意义。
