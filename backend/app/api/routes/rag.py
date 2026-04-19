from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import tempfile

from src.pdf_processor import process_pdf_to_images, get_file_hash
from src.llm_generator import generate_answer_with_vision

router = APIRouter()

# DTO schema for chat request
class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    chat_history: Optional[List[dict]] = None
    top_k: int = 3

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    接收 PDF 并处理为其页面缓存与特征。
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 格式文件")
        
    suffix = os.path.splitext(file.filename)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(prefix="rag_upload_", suffix=suffix, delete=False)
    temp_path = tmp.name
    tmp.close()
    try:
        # 将流保存到本地临时文件
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. 解析 PDF 到图像缓存
        image_paths = process_pdf_to_images(temp_path)
        
        if not image_paths:
            raise HTTPException(status_code=500, detail="未能成功解析出任何页面图像")
            
        file_hash = get_file_hash(temp_path)
        
        # 避免并发时引错库，延迟导入或从全局拿， 这里我们可以从全局获取 vector_store_instance，或者重新调单例
        # 为简单起见，从 main 导入全局实例
        from backend.app.main import vector_store_instance
        
        if vector_store_instance is None:
            raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")

        # 2. 将图片转换为 ColPali 向量并建立索引
        success = vector_store_instance.embed_and_store_documents(
            image_paths=image_paths,
            document_id=file_hash,
            document_name=file.filename
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="存储多模态特征库发生意外错误")

        return {
            "status": "success",
            "document_id": file_hash,
            "document_name": file.filename,
            "page_count": len(image_paths)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 无论成功错误，清除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/chat")
async def chat(req: ChatRequest):
    """
    接收对某几篇（或全部）文档的查询，
    进行两阶段召回后使用豆包生成答案，返回文字回答以及匹配到的页面图片（供溯源）。
    """
    try:
        from backend.app.main import vector_store_instance
        if vector_store_instance is None:
            raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")
            
        # 1. RAG 图像双路检索
        results = vector_store_instance.retrieve_with_two_stage(
            query_text=req.query,
            document_ids=req.document_ids,
            top_k=req.top_k
        )
        
        # 将被选中的匹配证据转换结构，准备喂给生成模型的同时发送前端
        evidence_images = []
        frontend_evidences = []
        
        for idx, res in enumerate(results):
            image_path = res["image_path"]
            doc_target = res["document_name"]
            page_idx = res["page_number"]
            # 给大模型做参考
            evidence_images.append(image_path)
            # 供展示给用户看的纯前端元数据结构
            frontend_evidences.append({
                "document_name": doc_target,
                "page_number": page_idx,
                "score": float(res["score"]),
                # 这里不传绝对路径，而是考虑我们可以发一个可以被图片直接读取的 URL 或者 base64。
                # 但考虑到图片可能在本地，我们可以先只把绝对路径当 src (本地调试时有权限)，其实正规做法应暴露静态文件路由。
                # 这里为了简单，我们会把图片转成 base64 给前端。
                "image_base64": image_path_to_base64(image_path)
            })

        if not evidence_images:
            return {
                "answer": "抱歉，向量库中未能检索出与该片段高度匹配的页面。",
                "evidences": []
            }
            
        # 2. 生成多模态 RAG 答案
        answer_text = generate_answer_with_vision(
            query_text=req.query,
            image_paths=evidence_images
        )
        
        return {
            "answer": answer_text,
            "evidences": frontend_evidences
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def image_path_to_base64(image_path: str) -> str:
    import base64
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    # 根据后缀决定 mimetype
    mime = "image/png"
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        mime = "image/jpeg"
    return f"data:{mime};base64,{encoded}"