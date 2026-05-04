# VisionDoc

![image-20260503161620235](assets/image-20260503161620235.png)

基于 **ColPali + MUVERA + Qdrant** 的视觉文档问答系统。上传 PDF、图片或 PPTX 文件，系统将每页统一渲染为图像后用 ColPali 建立视觉索引，查询时两阶段检索召回最相关页面，再由豆包视觉大模型生成答案，并附上原页截图供溯源。

## 🌟 核心特性

- **视觉文档支持**：PDF / PNG / JPG / JPEG / WEBP / PPTX，统一渲染为页面图像进入索引
- **两阶段检索**：MUVERA 压缩向量快速海选（Prefetch）+ ColPali 原始多向量 MaxSim 精准重排（Rerank）
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
├── requirements.txt
├── src/                           # AI 核心逻辑
│   ├── config.py                  # 环境变量（API Key、模型名、Qdrant URL 等）
│   ├── llm_generator.py           # 调用豆包视觉 LLM 生成答案
│   ├── doc_processor.py            # 视觉文档 → 页面图像缓存（PDF/图片/PPTX）
│   └── vector_store.py            # ColPali + MUVERA + Qdrant 封装
├── backend/                       # FastAPI 后端
│   ├── main.py                    # 应用入口，挂载静态前端
│   └── api/routes/
│       ├── health.py              # 健康检查
│       └── rag.py                 # 上传、对话、文件管理接口
└── frontend/                      # 前端客户端
    ├── index.html                 # 单页 HTML（Tailwind 样式）
    └── app.js                     # 所有交互逻辑
```

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
    -v $(pwd)/qdrant_local:/qdrant/storage:z \
    qdrant/qdrant
```

推荐把 Docker 版 Qdrant 作为默认运行路径。若 `QDRANT_URL` 不可达，后端会自动回退到仓库内的 `qdrant_local/` embedded 存储，方便本地开发或离线调试；但做 benchmark 时，建议先确认当前实例实际连接的是 Docker 暴露的 6333 端口。

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

1. 用户问题先经过轻量 guard，拦截明显与文档无关的闲聊 / 身份类问题
2. 如果问题包含多个由强分隔符拆出的独立子问题，则对子问题分别检索，并在后端合并证据页
3. 对单问题或每个子问题分别生成查询向量
4. 使用 MUVERA 在 Qdrant 中进行第一阶段 Prefetch
5. 用 ColPali 原始多向量执行 MaxSim 精排，得到最相关页面
6. 若有页面达到最小相关性阈值（默认 `score >= 0.60`），则直接使用这些证据页生成答案
7. 若 strict evidence 不足以填满当前证据窗口，后端会补入少量 near-threshold 页面，减少 raw 命中但 final 丢页的情况
8. 若某个问题或子问题连 near-threshold 页面也没有，但仍检索到候选页，则回退到该问题或子问题得分最高的 1 页证据继续生成答案；若页面内容已明确写出答案，可直接作答，否则会在说明层提示证据质量或写明无法确认
8. 将最终合并后的证据页截图与证据元数据送给豆包多模态模型生成答案

## 📏 证据质量分数说明

- 单页 `score`：Qdrant MaxSim 原始分除以 query token 数得到的归一化分数，目的是减弱查询长度对阈值的影响
- 顶部“证据质量”分数：当前已采用证据页中 top-3 分数的平均值，保留两位小数
- 这个分数是轻量相关性指标，不代表事实正确率；它更适合回答“当前回答依赖的证据页有多贴题”

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

> 这是一个以视觉理解和可追溯性为优先目标的多模态 RAG 原型。上传时要做文档转图和视觉嵌入，问答时要做两阶段检索和外部多模态生成，所以整体链路天然比纯文本 RAG 更重。机器性能会影响体验，但真正的优化方向应该基于分段时序数据来判断，而不是直接把问题归因为电脑弱。

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
    --output-dir eval_runs/first_pass
```

脚本会输出：

- `summary.json`：自动汇总的 retrieval / latency / citation 指标
- `detailed_results.json`：逐题原始结果，适合排查失败样例
- `review_sheet.csv`：带空白人工评分列的复核表，可继续补 `manual_answer_accuracy`、`manual_faithfulness`、`manual_citation_correctness`
- `benchmark_snapshot.json`：本次运行实际使用的 benchmark 快照，方便后续复现

推荐长期保留：`summary.json`、`review_sheet_reviewed.csv`、`review_summary.json`、`benchmark_snapshot.json`。
`detailed_results.json`、`review_sheet.csv`、`benchmarks/_validate_*` 和临时 smoke run 更适合在本地排查问题后清理掉，因此默认加入忽略规则。

如果你已经手工补完 `review_sheet.csv`，可以单独做一次人工评分汇总：

```bash
python scripts/run_rag_eval.py --review-csv eval_runs/first_pass/review_sheet.csv
```

`bootstrap_rag_benchmark.py` 会优先尝试基于单页文本自动生成“问题 + gold answer + gold page”的草稿；如果当前环境没有可用的大模型 key，也会退回到启发式模式，至少先帮你把文档页码、标题/重点句和 review CSV 搭起来，避免全量手工抄题。

第一版会自动统计：raw/final page Hit@1、Hit@3、MRR、doc Hit@K、fallback 使用率、unsupported sub-query 数、retrieval latency、首 token 延迟、keyword coverage、citation 是否命中 gold evidence。对于更可信的答案质量结论，仍建议结合人工评分结果一起使用。
