# VisionRAG-ColPali

多文档多模态 RAG 系统：使用 ColPali 直接理解 PDF 页面图像，结合 MUVERA + Qdrant 两阶段检索，再由 Doubao 视觉模型生成答案。

本仓库已进入分阶段升级状态：
- Phase 1 已完成：修复跨文档串档、证据重复、上传覆盖问题。
- Phase 2 已启动：新增 FastAPI + Next.js 脚手架，准备替换 Streamlit 入口。

## 推荐仓库名
- VisionRAG-ColPali

## 当前项目结构

```text
RAG/
├── app.py                         # 当前可运行入口（已支持多文档并存与范围检索）
├── requirements.txt
├── .env.example
├── backend/                       # Phase 2: FastAPI 脚手架
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       └── api/routes/health.py
├── frontend/                      # Phase 2: Next.js 脚手架
│   ├── package.json
│   ├── next.config.mjs
│   └── app/
│       ├── layout.tsx
│       ├── page.tsx
│       └── styles.css
├── data/
│   └── cache_images/
└── src/
    ├── config.py
    ├── pdf_processor.py
    ├── vector_store.py
    ├── llm_generator.py
    └── utils/helper.py
```

## 已完成的关键修复

1. 文档隔离入库
- 每页向量 payload 新增 document_id、document_name、page_number。
- 同一文档重复上传会先删除旧向量再写入，避免脏数据累计。

2. 检索范围控制
- 默认仅检索“当前文档”。
- 可切换为“全部文档”。

3. 证据去重
- 返回证据按 document_id + page_number + image_path 去重，避免 1 页文件机械重复显示。

4. 多文档并存
- 上传新 PDF 不会覆盖旧文档可见性，侧边栏可切换当前文档。

## 环境启动（当前可用）

### 1) Python 环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) 配置环境变量

```bash
cp .env.example .env
```

编辑 .env，设置 ARK_API_KEY。

### 3) 启动 Qdrant（建议持久化卷）

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4) 启动当前应用

```bash
streamlit run app.py
```

## Phase 2 脚手架试跑

### FastAPI

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Next.js

```bash
cd frontend
npm install
npm run dev
```

## 说明

- 当前生产可用路径仍是 Streamlit 入口。
- backend/frontend 已完成基础骨架，后续会把检索和入库能力完整迁移到 API + Web 前端。
