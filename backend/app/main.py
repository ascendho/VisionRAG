from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.api.routes.health import router as health_router

# 这个全局实例要在 lifespan 里初始化，之后在各个路由中导入
vector_store_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.pdf_processor import clear_all_caches
    from src.vector_store import VisionVectorStore

    # 启动时只做模型与向量库初始化，不主动清空业务数据，避免 --reload 模式下频繁重启导致数据丢失。
    global vector_store_instance
    if vector_store_instance is None:
        vector_store_instance = VisionVectorStore()

    print("✅ [Startup] 服务已准备就绪。")
    yield
    # 关闭时不清空向量库，避免热重启时误删。
    print("✅ [Shutdown] 服务已关闭。")

app = FastAPI(
    title="Gemini-Style VisionRAG Backend",
    version="1.0.0",
    description="Backend service for multi-document vision RAG with auto cache cleaning.",
    lifespan=lifespan
)

# 允许跨域请求，供 Next.js 前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)

# 放到最后导入防止出现循环依赖
from backend.app.api.routes.rag import router as rag_router
app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])

# 挂载前端静态文件，提供单一的 Gemini 风格纯 HTML/JS 服务
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
