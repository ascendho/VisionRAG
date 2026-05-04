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

## “人工复核草稿” 到底是什么

这里其实有两个不同阶段，容易混在一起：

1. **题目层的人工复核草稿**：指 `bootstrap_rag_benchmark.py` 生成 benchmark 草稿之后，你先人工扫一遍题目质量。
	- 主要文件：`benchmarks/rag_eval_small_draft.json`、`benchmarks/rag_eval_small_draft_review.csv`
	- 你要做的事：删掉跨页题、改短标准答案、核对页码和关键词
	- 目的：把“自动起草的候选题”清洗成“可以正式评测的 benchmark”
2. **回答层的人工评分表**：指 live evaluation 跑完后，再看模型每题答得对不对、证据稳不稳。
	- 主要文件：`data/eval_runs/<run_name>/review_sheet.csv`、`review_sheet_reviewed.csv`、`review_summary.json`
	- 你要做的事：补 `manual_answer_accuracy`、`manual_faithfulness`、`manual_citation_correctness`
	- 目的：给自动指标之外，再补一层人工质量判断

如果我前面说“人工复核草稿”，默认指的是第一种，也就是 benchmark 题目草稿的轻量人工清洗阶段，不是最终打分表。

## 产物保留建议

建议把 benchmark 相关产物分成“长期保留”和“临时排查”两类。

长期保留：

- `rag_eval_small_draft.json` 这类已经人工确认过的 benchmark 源文件。
- `sample-documents/` 中参与 benchmark 的源文档。
- 某次正式 run 下的 `summary.json`、`review_sheet_reviewed.csv`、`review_summary.json`、`benchmark_snapshot.json`。

临时排查后可删除：

- `benchmarks/_validate_bootstrap.json`
- `benchmarks/_validate_bootstrap_review.csv`
- 每次 run 中的 `detailed_results.json`
- 尚未补人工评分的 `review_sheet.csv`
- 只用于冒烟验证的 `data/eval_runs/smoke_*`

如果你希望仓库长期保持干净，建议只把 benchmark 定义文件纳入版本控制，把 `data/eval_runs/` 留在本地或单独归档。

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

## 这些指标是怎么算出来的

最容易混淆的是四组概念：

1. **raw vs final**
	- `raw_*` 看的是 `all_candidates`，也就是精排后的候选页列表。
	- `final_*` 看的是 `evidences`，也就是阈值筛选与 fallback 之后，真正送进回答生成的页面。
2. **page vs doc**
	- `page` 要求页码和文档都匹配。
	- `doc` 只要求文档匹配，不要求页码一致。
3. **单题值 vs summary 平均值**
	- 每一题先算出一个 `0/1` 或一个 MRR 值。
	- `summary.json` 再对所有题做平均，所以 `0.9` 本质上就是 20 题里有 18 题命中。
4. **自动指标 vs 人工评分**
	- `summary.json` 里这些是自动算出来的。
	- `manual_*` 只有在你补完 review sheet 之后才会出现。

一个最小例子：假设某题的 gold page 是 `waLLMartCache.pdf` 第 2 页，而检索返回：

```text
all_candidates = [p6, p5, p2]
evidences      = [p6, p5, p2]
```

那么：

- `raw_page_hit_at_1 = 0`，因为第 1 名是第 6 页，不是 gold page。
- `raw_page_hit_at_3 = 1`，因为前 3 个候选里出现了第 2 页。
- `raw_page_mrr = 1 / 3 = 0.3333`，因为第一个命中的 gold page 出现在第 3 名。
- 如果 `final` 列表和 `raw` 一样，那么 `final_page_hit_at_3` 和 `final_page_mrr` 也分别是 `1` 和 `0.3333`。

把这个逻辑放大到整份 benchmark：

- 如果 20 题里有 18 题 `final_page_hit_at_3 = 1`，2 题是 `0`，那么 `summary.json` 里的 `final_page_hit_at_3 = 18 / 20 = 0.9`。
- 如果 20 题里有 10 题 `fallback_evidence_used = 1`，另外 10 题是 `0`，那么 `fallback_rate = 10 / 20 = 0.5`。

回答相关的自动指标再看两步：

1. `answer_contains_gold_answer_rate`
	- 把回答文本和 `gold_answer` 都做归一化。
	- 如果标准答案字符串直接出现在回答里，这一题记 `1`，否则记 `0`。
2. `keyword_coverage_avg`
	- 看 `gold_answer_keywords` 里有多少关键词真的出现在回答中。
	- 例如关键词是 `['Redis', 'L2 cache']`，回答只命中了 `Redis`，那这一题 coverage 就是 `1 / 2 = 0.5`。

引用相关指标则是：

- `answer_has_citation_rate`：回答里有没有出现像 `[E3]` 这样的 evidence 引用。
- `citation_has_gold_evidence_rate`：这些被引用的 evidence 里，是否至少有一个真正命中了 gold page。

所以，当前 20 题结果里的 `final_page_hit_at_3 = 0.9` 可以直译成：20 题里有 90% 的问题，在最终真正用于回答的证据页前 3 名中，包含了正确 gold page。

## 10 题起步时的题型建议

如果你先做 10 题，一个足够实用的分布通常是：

- 3 到 4 道数值/事实查找题。
- 3 到 4 道定义/单页总结题。
- 2 到 3 道列表、约束或步骤类问题。
- 1 到 2 道相对复杂的问题。

如果当前两份 PDF 里没有明显图表页，就不要强行凑图表题；先把单页可判分的题做稳更重要。