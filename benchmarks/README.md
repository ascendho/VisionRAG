# RAG Benchmark 使用说明

这个目录用于放置小规模、高可信、可复核的评测集，目标不是一次做很大，而是先产出一组能稳定复现、能写进简历的 RAG 指标。

## 推荐起步范围

如果你刚开始做这一轮实验，建议先收敛到：

1. 2 到 3 份已经完成索引的真实文档。
2. 10 道可以明确判分的问题。
3. 每题只绑定 1 个主 gold page；如果确实有多个可接受证据页，再补到 `gold_evidence` 列表里。

对当前仓库，第一轮最稳的做法通常是先用 2 份 PDF 起步，再视需要补 1 份 PPTX。

## 不想全手工标注怎么办

仓库现在提供了一条“先自动起草，再人工轻量复核”的流程：

```bash
python scripts/bootstrap_rag_benchmark.py \
	--documents waLLMartCache.pdf MemGPT.pdf \
	--target-question-count 10 \
	--output-file benchmarks/rag_eval_small_draft.json
```

这个脚本会做几件事：

1. 直接读取 `benchmarks/sample-documents/` 目录里的 PDF / PPTX 源文件。
2. 为每个文档抽取单页文本，并优先选择信息量更高的页面。
3. 如果当前环境有可用的大模型 key，就尝试基于单页文本自动生成“问题 + 标准答案 + 关键词 + gold page”草稿。
4. 如果没有可用的大模型 key，就退回到启发式模式，至少先帮你生成页码锚点、标题/重点句型问题和 review CSV，减少抄录工作量。
5. 同时输出一份 review CSV，方便你快速修改问题、答案和备注。

注意：自动起草只能降低成本，不能完全替代人工复核。最终至少要确认每题的页码、问题是否单页可答、答案是否足够短且不歧义。

## 推荐工作流

1. 先上传并完成索引 2 到 3 份目标文档。
2. 用 [rag_eval_template.json](rag_eval_template.json) 看 schema，或者直接用 `bootstrap_rag_benchmark.py` 生成草稿 benchmark。
3. 检查自动生成的 JSON 和 review CSV，把明显歧义、跨页依赖或答案过长的问题删掉或改掉。
4. 用 [scripts/run_rag_eval.py](../scripts/run_rag_eval.py) 对真实 `/api/rag/chat` 跑批量评测。
5. 补完 `review_sheet.csv` 里的人工评分列。
6. 再跑一次 review summary，得到答案准确性、faithfulness、citation correctness 的汇总结果。

## Benchmark 文件结构

顶层字段：

- `version`：schema 版本。
- `description`：这份 benchmark 的说明，比如使用了哪些文档、目标是什么。
- `defaults`：默认 `top_k`、`min_score`、`chat_history` 等配置。
- `items`：真正参与评测的问题列表。

每个 `item` 至少要包含：

- `id`：稳定的问题编号。
- `question`：发给 `/api/rag/chat` 的实际问题。
- `question_type`：题型标签，例如 `numeric_fact`、`definition_summary`、`single_page_fact`、`title_lookup`。
- `gold_evidence`：可接受证据页列表。
- `gold_answer`：标准答案，尽量保持一句话或几个短语。

常用可选字段：

- `document_ids`：把检索范围限制到特定文档。
- `gold_answer_keywords`：答案里应该出现的关键词或数字。
- `notes`：给后续复核者或自己看的说明。
- `source_excerpt`：自动起草时附带的页内文本摘录，便于复核。

## Gold Evidence 格式

每个 `gold_evidence` 条目建议包含：

- `page_number`：必填，必须是正整数。
- `document_id`：强烈建议填写。当前项目上传时默认用文件内容哈希作为 `document_id`，自动起草脚本会直接按这个规则填充。
- `document_name`：可作为 `document_id` 的兜底显示信息。

只要能用 `document_id`，就优先不要只靠文件名匹配。文件名重复或后续重命名时，`document_id` 更稳。

## 人工复核建议

评测脚本会生成一个带空白列的 `review_sheet.csv`。第一轮只有 10 题时，推荐直接手工补这三列：

- `manual_answer_accuracy`：`1` 正确，`0.5` 部分正确，`0` 错误。
- `manual_faithfulness`：`1` 完全被返回证据支撑，`0.5` 部分支撑或支撑较弱，`0` 明显不被证据支撑。
- `manual_citation_correctness`：`1` 引用页正确，`0.5` 有引用但不完整或部分错误，`0` 没引用或引用错误。

如果你只做 10 题，这部分人工成本并不高，但可信度会比完全依赖自动 judge 高得多。

## 当前脚本会自动统计什么

- `all_candidates` 上的 raw page Hit@1、Hit@3、MRR。
- threshold 后 `evidences` 上的 final page Hit@1、Hit@3、MRR。
- document-level Hit@1、Hit@3。
- fallback 使用率和 unsupported sub-query 数。
- retrieval latency、首 token 延迟、整轮对话延迟。
- `gold_answer_keywords` 的覆盖率。
- 回答中引用的 evidence 是否命中 gold page。

## 10 题起步时的题型建议

如果你先做 10 题，一个足够实用的分布通常是：

- 3 到 4 道数值/事实查找题。
- 3 到 4 道定义/单页总结题。
- 2 到 3 道列表、约束或步骤类问题。
- 1 到 2 道相对复杂的问题。

如果当前两份 PDF 里没有明显图表页，就不要强行凑图表题；先把单页可判分的题做稳更重要。