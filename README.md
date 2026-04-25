# VisionRAG

![image-20260425145610310](assets/image-20260425145610310.png)

基于 **ColPali + MUVERA + Qdrant** 的多格式视觉文档问答系统。上传 PDF、图片或纯文本文件，系统将每页渲染为图像后用 ColPali 建立视觉索引，查询时两阶段检索召回最相关页面，再由豆包视觉大模型生成答案，并附上原页截图供溯源。

前端采用纯 Vanilla HTML/JS + Tailwind CSS 构建，风格参考 Gemini；后端基于 FastAPI 提供服务。

## 🌟 核心特性

- **多格式文件支持**：PDF / PNG / JPG / JPEG / WEBP / TXT / MD，统一渲染为页面图像进入索引
- **两阶段检索**：MUVERA 压缩向量快速海选（Prefetch）+ ColPali 原始多向量 MaxSim 精准重排（Rerank）
- **归一化相关性得分**：MaxSim 原始分除以查询 token 数，得分落在 0–1 区间，阈值稳定不受查询长度影响
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
| 图像处理 | Pillow（图片缓存、文本渲染）|

## 📂 项目结构

```text
RAG/
├── requirements.txt
├── src/                           # AI 核心逻辑
│   ├── config.py                  # 环境变量（API Key、模型名、Qdrant URL 等）
│   ├── llm_generator.py           # 调用豆包视觉 LLM 生成答案
│   ├── doc_processor.py            # 多格式文档 → 页面图像缓存（PDF/图片/文本）
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

> macOS 需要安装 Poppler（PDF 解析依赖）：
> ```bash
> brew install poppler
> ```

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

### 4. 运行服务

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

打开浏览器访问 [http://localhost:8000](http://localhost:8000)

## 📖 使用说明

1. **上传文件**：点击左侧上传按钮，支持 PDF、PNG、JPG、WEBP、TXT、MD 格式
2. **多文档管理**：点击右上角文件图标查看已加载文档，可下载原文件或删除
3. **作用域选择**：上传 2 个及以上文档后，输入框上方出现 Scope Bar，可点击选中特定文档进行对话
4. **提问**：在输入框输入问题，回车或点击发送；回答下方附有相关页面截图，点击可全屏查看

## ⚙️ 主要 API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/rag/upload` | 上传并索引文件 |
| `POST` | `/api/rag/chat` | 发起对话查询 |
| `GET` | `/api/rag/files` | 列出所有已索引文档 |
| `GET` | `/api/rag/files/{id}/download` | 下载原始文件 |
| `DELETE` | `/api/rag/files/{id}` | 删除文档及其索引 |
| `GET` | `/api/health` | 服务健康检查 |

## 📊 性能 Benchmark

`eval/` 目录提供三个独立的性能测试脚本，**无需 gold-page 标注**，直接测量系统延迟与吞吐。

### 冷热缓存定义

| track | cold | hot |
|---|---|---|
| 上传 | 页面图像缓存已清除（首次上传） | 页面图像缓存已预热 |
| 问答 | 第 1 次遍历查询集（Qdrant 页表冷） | 第 2+ 次遍历查询集 |

### 上传链路 benchmark

```bash
python -m eval.upload_benchmark \
  --input qdrant_local/pdfs/your.pdf \
  --mode both \
  --runs 3
```

关键指标：`pdf_render_ms`、`embedding_ms`、`upsert_ms`、`total_index_ms`

### 问答链路 benchmark

```bash
python -m eval.answer_benchmark \
  --queries eval/queries/answer_queries.jsonl \
  --runs 3
```

可选参数：`--top-k`（默认 5）、`--prefetch-multiplier`（默认 10）、`--min-score`（默认 0.6）、`--max-tokens`（默认 800）

关键指标：`time_to_evidence_ms`、`time_to_first_token_ms`、`total_answer_latency_ms`、`tokens_per_second`

### 架构消融实验

单变量扫描，第一个值为 baseline：

```bash
# 问答 track：扫描 prefetch_multiplier
python -m eval.architecture_ablation \
  --track answer \
  --knob prefetch_multiplier \
  --values "1,5,10,20,50" \
  --queries eval/queries/answer_queries.jsonl \
  --runs 3

# 上传 track：扫描 DPI
python -m eval.architecture_ablation \
  --track upload \
  --knob dpi \
  --values "75,150,300" \
  --input qdrant_local/pdfs/your.pdf \
  --runs 3
```

upload 支持的 knob：`dpi`、`embed_batch_size`、`upsert_batch_size`、`muvera_r_reps`、`muvera_dim_proj`

answer 支持的 knob：`prefetch_multiplier`、`top_k`、`min_score`、`max_tokens`

所有结果以 JSONL 格式保存在 `eval/results/` 目录。
