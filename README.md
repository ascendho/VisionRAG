# VisionRAG

多文档多模态 RAG 系统：使用 ColPali 直接理解 PDF 页面图像，结合 MUVERA + Qdrant 两阶段检索，再由大型视觉模型（如 Doubao、Gemini）生成答案。

当前版本拥有一个现代化的类 Gemini 风格客户端，采用纯 Vanilla HTML/JS+Tailwind 构建前端，基于 FastAPI 提供强劲响应。

## 🌟 核心特性
- **支持多文件上传与管理**: 并发特征提取保存至 Qdrant，支持查看当前已加载的所有文档 (`/api/rag/files`)
- **快捷键交互**: 回车键 (`Enter`) 换行，命令+回车 (`Cmd+Enter` / `Ctrl+Enter`) 发送消息
- **多模态精准溯源**: 提供原文页面图片引用预览，点击图片以弹窗大图形式查看
- **极简式无缝 UI**: 精致的欢迎页网格导航，通过右上角快速查询已加载文件，纯粹顺滑的使用体验

## 📂 项目结构

```text
RAG/
├── fix_ui.py                      # UI 更新补丁或辅助脚本
├── requirements.txt
├── src/                           # AI 核心逻辑引擎
│   ├── config.py
│   ├── llm_generator.py           # 大模型生成逻辑
│   ├── pdf_processor.py           # PDF 解析
│   └── vector_store.py            # Qdrant + ColPali + Muvera 封装
├── backend/                       # FastAPI 后端服务
│   ├── main.py                # FastAPI 启动文件及静态资源代理
│   └── api/routes/
│       ├── health.py          # 健康检查
│       └── rag.py             # RAG 对话、文件上传与已加载文档查询
├── frontend/                      # 现代 UI 客户端
│   ├── index.html                 # 纯前段 HTML/Tailwind 页面
│   └── app.js                     # 纯前段 JS 逻辑
└── qdrant_local/                  # 本地向量库及上传文件缓存
```

## 🚀 快速启动

**1. 准备环境 (推荐 Python 3.10+)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. 启动基础依赖 (Qdrant)**
确保本地已安装运行 Qdrant 数据库（如使用 Docker）：
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_local:/qdrant/storage:z \
    qdrant/qdrant
```

**3. 运行服务并访问**
使用 uvicorn 跑起 FastAPI 后端，会自动代理挂载 frontend 静态文件夹。
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
打开浏览器访问 [http://localhost:8000/](http://localhost:8000/) 即可体验前端最新版本。
