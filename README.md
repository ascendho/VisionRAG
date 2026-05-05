# 页证智答

![image-20260503161620235](assets/image-20260503161620235.png)

**ColPali + MUVERA + Qdrant** 驱动的多模态文档检索与问答系统。上传 PDF、图片或 PPTX 文件后，系统会将每页统一渲染为图像并建立视觉索引；当前网页端默认沿用 MUVERA Prefetch + ColPali Rerank 的两阶段检索路径，再由豆包多模态大模型生成答案，并附上原页截图供溯源。最新 20 题 benchmark 显示：两阶段路径在当前小规模语料上能够稳定取得 `90% final page Hit@3` 与 `0.7667 final page MRR`，也体现出“轻量召回 + 精排复核”的双路检索设计思路；但在当前 3 文档配置下，单阶段 ColPali 的平均检索延迟更低，因此两阶段路径更适合表述为具备工程扩展潜力的检索架构，而不是已被当前实验充分证明的最优默认策略。

## 🌟 核心特性

- **视觉文档支持**：PDF / PNG / JPG / JPEG / WEBP / PPTX，统一渲染为页面图像进入索引
- **可切换检索架构**：支持 `two_stage`（MUVERA Prefetch + ColPali Rerank）、`colpali_only` 和 `muvera_only` 三种模式；当前网页端默认走 `two_stage`，其主要价值在于把轻量召回与视觉精排解耦，为更大规模语料预留扩展空间；在最新 20 题 benchmark 上，`colpali_only` 在保持相同检索质量时平均更快
- **归一化相关性得分**：MaxSim 原始分除以查询 token 数，得分落在 0–1 区间；默认采用 `score >= 0.60` 作为正式证据阈值，并补入少量 near-threshold 页面，减少 final evidence 丢页
- **复合问题支持**：对以强分隔符拆出的多个子问题分别检索，再合并证据页，减少多问合并时只覆盖单一主题的问题
- **多文档作用域查询**：Scope Bar 支持同时选中多个文档进行对话，也可切回全库检索
- **原页图片溯源**：每条回答附带匹配页面缩略图网格（最多 5 列自适应），点击大图查看；卡片显示文档名、页码与相关性得分
- **现代化 UI**：欢迎页 → 聊天页平滑过渡，暗色模式，Markdown 渲染，一键复制回答

## 🛠️ 技术栈

| 层次 | 技术 |
|---|---|
| 视觉嵌入模型 | ColPali v1.3（`vidore/colpali-v1.3`），128-D 多向量 |
| 加速检索 | MUVERA 16-D FDE（fastembed），Prefetch 倍率 5× |
| 向量数据库 | Qdrant（本地 Docker 或云端），MaxSim 比较器 |
| 大语言模型 | 豆包 Seed 2.0 Pro（`doubao-seed-2-0-pro-260215`），Volcengine ARK API |
| 后端框架 | FastAPI + uvicorn |
| 前端 | 原生 HTML/JS + Tailwind CSS CDN + marked.js |
| PDF 解析 | pdf2image + Poppler |
| 文档转图 | pdf2image + Poppler；PPTX 通过 LibreOffice `soffice` 先转 PDF |
| 图像处理 | Pillow（图片缓存与标准化）|

## 📂 项目结构

```text
RAG/
├── backend/                       # FastAPI 后端
│   ├── main.py                    # 应用入口，挂载静态前端
│   └── api/routes/
│       ├── health.py              # 健康检查
│       └── rag.py                 # 上传、对话、文件管理接口
├── benchmarks/                    # benchmark 定义与样本文档
│   ├── README.md
│   ├── rag_eval_small_draft.json
│   └── sample-documents/
├── data/                          # 本地运行产物（默认忽略）
│   ├── eval_runs/                 # 评测结果输出
│   └── qdrant/                    # embedded Qdrant 数据与原始上传文件
├── frontend/                      # 前端客户端
│   ├── index.html                 # 单页 HTML（Tailwind 样式）
│   └── app.js                     # 所有交互逻辑
├── scripts/                       # benchmark 起草与评测脚本
│   ├── bootstrap_rag_benchmark.py
│   ├── build_retrieval_ablation_summary.py
│   └── run_rag_eval.py
├── src/                           # AI 核心逻辑
│   ├── config.py                  # 环境变量、运行目录与模型配置
│   ├── llm_generator.py           # 调用豆包视觉 LLM 生成答案
│   ├── doc_processor.py           # 视觉文档 → 页面图像缓存（PDF/图片/PPTX）
│   └── vector_store.py            # ColPali + MUVERA + Qdrant 封装
├── tests/                         # 回归测试
├── assets/                        # README 配图等静态资源
├── README.md
└── requirements.txt
```

从这一版开始，仓库会把持久化运行数据统一放到 `data/` 下；如果根目录里还残留旧的 `qdrant_local/` 或 `eval_runs/`，首次运行配置或评测脚本时会自动迁移到新位置。

## 🚀 快速启动

### 1. 准备环境（推荐 Python 3.10+）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> macOS 需要安装 Poppler 和 LibreOffice：
> ```bash
> brew install poppler
> brew install --cask libreoffice
> ```
>
> PPTX 支持依赖 LibreOffice 提供的 `soffice` 转换能力；如果本机没有该命令，上传 PPTX 会直接返回可操作的错误提示。

### 2. 配置环境变量

在项目根目录创建 `.env` 文件（或直接在 `src/config.py` 中设置）：

```env
ARK_API_KEY=your_volcengine_ark_api_key
DOUBAO_MODEL_NAME=doubao-seed-2-0-pro-260215
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=colpali-rag-collection
```

### 3. 启动 Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/data/qdrant:/qdrant/storage:z \
    qdrant/qdrant
```

推荐把 Docker 版 Qdrant 作为默认运行路径。若 `QDRANT_URL` 不可达，后端会自动回退到仓库内的 `data/qdrant/` embedded 存储，方便本地开发或离线调试；但做 benchmark 时，建议先确认当前实例实际连接的是 Docker 暴露的 6333 端口。

### 4. 运行服务

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

打开浏览器访问 [http://localhost:8000](http://localhost:8000)

## 📖 使用说明

1. **上传文件**：点击左侧上传按钮，支持 PDF、PNG、JPG、WEBP、PPTX 格式
2. **多文档管理**：点击右上角文件图标查看已加载文档，可下载原文件或删除
3. **作用域选择**：上传 2 个及以上文档后，输入框上方出现 Scope Bar，可点击选中特定文档进行对话
4. **提问**：在输入框输入问题，回车或点击发送；回答下方附有相关页面截图，点击可全屏查看

## 🔄 系统运行流程

### 1. 上传与建索引

1. 接收 PDF / 图片 / PPTX 文件
2. 将文档统一转换为页面图像
    - PPTX 会先转成 PDF，再复用现有 PDF 转页图链路
3. 使用 ColPali 为每一页生成多向量视觉特征
4. 使用 MUVERA 生成压缩向量，作为第一阶段快速召回特征
5. 将原始多向量与压缩向量一起写入 Qdrant

### 2. 对话与回答

1. 用户问题先经过轻量规则 guard，拦截明显与文档内容无关的身份类或闲聊问题，例如“你是谁”“讲个笑话”；这一阶段只做快速规则匹配，不额外调用 LLM，避免无效检索开销
2. 如果问题包含多个由 `？`、`?`、`；`、`;` 或换行拆出的独立子问题，后端会先做多子问题拆分与查询规划，再对子问题分别检索并在后端合并证据页；例如“文档提到了哪些人物？他们分别负责什么；结论是什么？”会被拆成多个子问题分别检索，减少多问合并时只覆盖单一主题的问题
    - 若后续子问题出现“他 / 它 / 那它”这类指代，后端会优先桥接前一个子问题的局部上下文，再进入后续检索改写逻辑
3. 当前网页端默认不显式传 `retrieval_mode`，后端会按 `two_stage` 路径处理；评测脚本可切换到 `colpali_only` / `muvera_only` 做检索消融
4. 对单问题或每个子问题分别生成查询向量
5. 默认路径会先使用 MUVERA 在 Qdrant 中进行第一阶段 Prefetch，再用 ColPali 原始多向量执行 MaxSim 精排；当前 20 题小规模 benchmark 尚未显示该路径优于 `colpali_only`
6. 若有页面达到最小相关性阈值（默认 `score >= 0.60`），则直接使用这些证据页生成答案
7. 若 strict evidence 不足以填满当前证据窗口，后端会补入少量 near-threshold 页面，减少 raw 命中但 final 丢页的情况
8. 若某个问题或子问题连 near-threshold 页面也没有，但仍检索到候选页，则回退到该问题或子问题得分最高的 1 页证据继续生成答案；若页面内容已明确写出答案，可直接作答，否则会在说明层提示证据质量或写明无法确认
9. 将最终合并后的证据页截图与证据元数据送给豆包多模态模型生成答案

## 📏 证据质量与 benchmark 指标说明

- 单页 `score`：Qdrant MaxSim 原始分除以 query token 数得到的归一化分数，目的是减弱查询长度对阈值的影响
- 顶部“证据质量”分数：当前已采用证据页中 top-3 分数的平均值，保留两位小数
- 这个分数是轻量相关性指标，不代表事实正确率；它更适合回答“当前回答依赖的证据页有多贴题”

常用 benchmark 指标可以这样理解：

- `final_page_hit_at_3`：最终返回给用户的前 3 个证据页中，是否至少出现 1 个 gold page。它回答“正确页有没有被找回来”。
- `final_page_mrr`：gold page 在最终证据排序中的平均倒数排名。命中越靠前，分数越高；第 1 位是 `1.0`，第 2 位是 `0.5`，第 3 位约为 `0.33`。
- `citation_has_gold_evidence_rate`：回答里被引用的证据页中，是否至少有 1 页真正命中 gold evidence 的比例。它回答“模型引用得对不对”。
- `answer_contains_gold_answer_rate`：生成答案中是否直接包含 gold answer 字符串的比例，也可以把它理解为 gold answer 覆盖率。它回答“标准答案有没有被明确说出来”。
- `automatic_faithfulness`：多模态 judge 基于“问题 + 回答 + 返回证据页图片”打出的支撑度分数。它衡量的是“回答是否被返回证据直接支持”，不是绝对事实正确率；当前 judge 使用 `>= 0.85 / >= 0.35 / < 0.35` 三档，对应 `supported / partially_supported / unsupported`。

## ⏱️ 性能与耗时说明

当前项目是多模态视觉 RAG，慢主要来自三类成本，而不是单一组件瓶颈：

1. **文档预处理**：PDF 转图、PPTX 转 PDF 再转图，本身就有 IO 与图像处理开销
2. **视觉嵌入**：ColPali 要对每页图像做多向量推理，这是上传阶段最重的本地计算
3. **最终回答生成**：检索完成后还要调用豆包多模态模型生成答案，这通常比 Qdrant 查询更慢

项目当前已在日志中输出如下时序：

- 上传阶段：`document_render_ms`、`embedding_ms`、`point_build_ms`、`qdrant_upsert_ms`、`total_index_ms`
- 检索阶段：`query_embedding_ms`、`qdrant_query_ms`、`result_format_ms`、`total_retrieval_ms`
- 生成阶段：`first_token_ms`、`total_generation_ms`

如果面试官问“为什么慢，是不是电脑性能太弱”，更准确的回答是：

> 这是一个以视觉理解和可追溯性为优先目标的多模态 RAG 原型。上传时要做文档转图和视觉嵌入，问答时当前默认走两阶段检索路径，并调用外部多模态模型生成答案，所以整体链路天然比纯文本 RAG 更重。机器性能会影响体验，但真正的优化方向应该基于分段时序数据来判断，而不是直接把问题归因为电脑弱。

现实中的产品之所以通常不会让用户感到这么慢，核心不是只靠更强的机器，而是靠系统设计把重活拆开：

1. **异步索引**：上传接口已改为先秒回 `task_id`，后台再做转图、嵌入和入库，并通过 SSE 推送阶段进度
2. **分层检索**：优先走更轻的文本索引或缓存命中，只有必要时才回退到更重的视觉检索
3. **更轻的在线模型**：先用更快的模型给出首答或首 token，再决定是否调用更重模型补全高保真答案
4. **缓存与预计算**：对热门文档、热门问题、文档文本层和页面特征做预热，尽量避免重复计算

所以，真实产品“感觉很快”的关键在于：用户不需要同步等待整条最重的视觉 RAG 链路跑完。

## ⚙️ 主要 API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/rag/upload` | 上传文件并立即返回 `task_id`，后台异步索引 |
| `GET` | `/api/rag/jobs/{task_id}` | 查询上传任务当前状态 |
| `GET` | `/api/rag/jobs/{task_id}/events` | 通过 SSE 订阅上传任务进度 |
| `POST` | `/api/rag/chat` | 发起对话查询 |
| `GET` | `/api/rag/files` | 列出所有已索引文档 |
| `GET` | `/api/rag/files/{id}/download` | 下载原始文件 |
| `DELETE` | `/api/rag/files/{id}` | 删除文档及其索引 |
| `GET` | `/api/health` | 服务健康检查 |

## 📊 RAG 评测实验

仓库里已经提供了一版基于真实 `/api/rag/chat` 接口的离线评测 harness，可直接用来跑小规模 benchmark、统计检索指标，并导出人工复核表。

推荐工作流：

1. 第一轮先收敛到 2 到 3 份真实 PDF / PPTX 文档、10 道题，并确认后端已经完成索引。
2. 如果不想从零手工写题，可以先运行自动起草脚本，批量生成一个待复核 benchmark 草稿：

```bash
python scripts/bootstrap_rag_benchmark.py \
    --documents waLLMartCache.pdf MemGPT.pdf \
    --target-question-count 10 \
    --output-file benchmarks/rag_eval_small_draft.json
```

3. 再参考 [benchmarks/README.md](benchmarks/README.md) 和 [benchmarks/rag_eval_template.json](benchmarks/rag_eval_template.json) 做最后复核。
4. 启动后端服务后运行：

```bash
python scripts/run_rag_eval.py \
    --benchmark-file benchmarks/rag_eval_small_draft.json \
    --api-base-url http://127.0.0.1:8000 \
    --output-dir data/eval_runs/first_pass
```

如果要做检索模式消融，可在同一份 benchmark 上分别追加 `--retrieval-mode two_stage`、`--retrieval-mode colpali_only`、`--retrieval-mode muvera_only`，再用 `scripts/build_retrieval_ablation_summary.py` 合并三份 `summary.json`。

当前 live eval 默认还会额外调用一次多模态 judge，生成 `automatic_faithfulness` 分数；如果只想保留旧版评测链路、跳过这部分额外开销，可在命令后追加 `--disable-automatic-faithfulness`。

### 关于 automatic faithfulness 的解释

`automatic_faithfulness` 衡量的是“返回给模型的证据页，是否能直接支撑最终回答”，而不是“答案在所有外部事实维度上是否绝对正确”。因此：

- 如果回答完全被返回证据页支持，它可以得到 `1.0`；
- 这不等于系统整体已经满分，仍需要结合 `answer_contains_gold_answer_rate`、`citation_has_gold_evidence_rate` 和人工复核一起解读；
- 在当前最新版 20 题 run 中，`automatic_faithfulness_avg = 1.00` 且 `automatic_faithfulness_scored_samples = 20`，更适合理解为“回答对已返回证据的依赖非常强、幻觉较少”，而不是“系统在所有维度都已经达到 100% 正确”。

脚本会输出：

- `summary.json`：自动汇总的 retrieval / latency / citation 指标
- `detailed_results.json`：逐题原始结果，适合排查失败样例
- `review_sheet.csv`：带空白人工评分列的复核表，可继续补 `manual_answer_accuracy`、`manual_faithfulness`、`manual_citation_correctness`
- `benchmark_snapshot.json`：本次运行实际使用的 benchmark 快照，方便后续复现

推荐长期保留：`summary.json`、`review_sheet_reviewed.csv`、`review_summary.json`、`benchmark_snapshot.json`。
`detailed_results.json`、`review_sheet.csv`、`benchmarks/_validate_*` 和临时 smoke run 更适合在本地排查问题后清理掉，因此默认加入忽略规则。

如果你已经手工补完 `review_sheet.csv`，可以单独做一次人工评分汇总：

```bash
python scripts/run_rag_eval.py --review-csv data/eval_runs/first_pass/review_sheet.csv
```

`bootstrap_rag_benchmark.py` 会优先尝试基于单页文本自动生成“问题 + gold answer + gold page”的草稿；如果当前环境没有可用的大模型 key，也会退回到启发式模式，至少先帮你把文档页码、标题/重点句和 review CSV 搭起来，避免全量手工抄题。

第一版会自动统计：raw/final page Hit@1、Hit@3、MRR、doc Hit@K、fallback 使用率、unsupported sub-query 数、retrieval latency、首 token 延迟、keyword coverage、citation 是否命中 gold evidence，以及基于证据页图片的 `automatic_faithfulness`。对于更可信的答案质量结论，仍建议结合人工评分结果一起使用。

### 当前 20 题 benchmark 结果

当前正式 benchmark 为 20 题，覆盖 3 份真实 PDF / PPTX 文档。最近一轮结果保存在 `data/eval_runs/faithfulness_20q_20260505_150958/summary.json`，其中：

- `final_page_hit_at_3 = 0.90`
- `final_page_mrr = 0.7667`
- `citation_has_gold_evidence_rate = 0.90`
- `answer_contains_gold_answer_rate = 0.80`
- `keyword_coverage_avg = 0.90`
- `automatic_faithfulness_avg = 1.00`
- `automatic_faithfulness_scored_samples = 20`
- `fallback_rate = 0.50`

这组数字属于自动 benchmark 指标，不是人工复核分数。需要强调的是，`automatic_faithfulness = 1.00` 反映的是“返回证据是否直接支撑回答”；同一轮 `answer_contains_gold_answer_rate = 0.80` 也说明系统整体答案正确率并未达到满分。当前 20 题 run 还没有补完人工评分；已经完成人工复核的是 earlier 10 题 reviewed subset，对应 `manual_answer_accuracy_avg = 1.0`、`manual_faithfulness_avg = 1.0`、`manual_citation_correctness_avg = 0.95`。

### 当前 20 题检索消融结果

最新检索模式消融结果保存在 `data/eval_runs/retrieval_ablation_20q_20260504_152822/`。在同一份 20 题 benchmark 上：

- `two_stage`: `raw_page_hit_at_3 = 0.90`，`raw_page_mrr = 0.7792`，`avg_total_retrieval_ms = 444.327`
- `colpali_only`: `raw_page_hit_at_3 = 0.90`，`raw_page_mrr = 0.7792`，`avg_total_retrieval_ms = 229.311`
- `muvera_only`: `raw_page_hit_at_3 = 0.45`，`raw_page_mrr = 0.3333`，`avg_total_retrieval_ms = 416.154`

这意味着：双路检索通过“MUVERA 轻量召回 + ColPali 精排复核”的分工设计，在工程上兼顾了可扩展性与页级定位质量。相较 `muvera_only`，它显著提升了页级命中与排序质量；但在当前 3 文档的小规模语料下，`colpali_only` 仍能以更低延迟达到相同质量。因此，当前实验更适合把两阶段检索写成“具有架构设计优势和扩展潜力的双路检索方案”，而不是“已在现配置下全面优于单阶段 ColPali 的默认策略”。

### 简历 / 项目介绍表述参考

如果简历里只保留 3 到 4 个指标，当前更推荐：`final_page_hit_at_3 = 90%`、`final_page_mrr = 0.7667`、`citation_has_gold_evidence_rate = 90%`、`answer_contains_gold_answer_rate = 80%`。`automatic_faithfulness = 1.00（20/20 scored）` 更适合作为补充说明，用来表达“回答基本被返回证据直接支持”，不建议单独作为 headline。

中文：实现融合 ColPali、MUVERA 与 Qdrant 的多模态视觉 RAG 系统，支持 PDF / PPTX 文档问答、页面级证据回溯、复合问题检索与多轮上下文改写；搭建 20 题、覆盖 3 份真实文档的 benchmark，在自动评测中取得 90% final page Hit@3、0.7667 final page MRR、90% 引用命中 gold evidence 与 80% gold answer 覆盖率，并补入 20/20 scored 的 automatic faithfulness 指标，用于衡量答案是否被返回证据直接支持。

English: Built a multimodal visual RAG system on top of ColPali, MUVERA, and Qdrant for PDF/PPTX question answering with page-level evidence grounding and compound-query retrieval; built a 20-question benchmark across 3 real documents, achieved 90% final page Hit@3, 0.7667 final page MRR, 90% citation-to-gold-evidence match, and 80% gold-answer coverage in automated evaluation, and added a 20/20-scored automatic faithfulness metric to measure whether answers are directly supported by returned evidence pages.

如果你想把人工复核结论也写进去，建议单独注明样本范围，例如：在 earlier reviewed 10-question subset 上，manual answer accuracy 和 faithfulness 均为 1.0，manual citation correctness 为 0.95。这样不会把 10 题人工结果误说成 20 题人工结果。
