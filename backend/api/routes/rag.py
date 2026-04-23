from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import shutil
import tempfile
import glob

from src.doc_processor import process_pdf_to_images, process_image_to_images, process_text_to_images, get_file_hash
from src.llm_generator import generate_answer_stream

router = APIRouter()

# DTO schema for chat request
class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    chat_history: Optional[List[dict]] = None
    top_k: int = 5
    min_score: float = 0.6


def _rebuild_image_cache_if_needed(results: list) -> None:
    """
    检查检索结果中的图像缓存文件是否存在（系统重启后 /tmp 会被清空）。
    若缺失，则从 qdrant_local/pdfs/ 中找到对应 PDF 并重新渲染图像缓存。
    """
    missing_doc_ids: dict = {}
    for r in results:
        image_path = str(r.get("image_path", ""))
        if image_path and not os.path.exists(image_path):
            doc_id = r.get("document_id", "")
            if doc_id and doc_id not in missing_doc_ids:
                missing_doc_ids[doc_id] = r.get("document_name", doc_id)

    if not missing_doc_ids:
        return

    pdfs_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
    for doc_id, doc_name in missing_doc_ids.items():
        matches = glob.glob(os.path.join(pdfs_dir, f"{doc_id}_*"))
        file_path = matches[0] if matches else None
        if not file_path or not os.path.exists(file_path):
            print(f"[Warning] File not found for document '{doc_name}' ({doc_id}), cannot rebuild image cache.")
            continue
        print(f"[Info] Rebuilding image cache for: {doc_name}")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".pdf":
                process_pdf_to_images(file_path)
            elif ext in {".txt", ".md"}:
                process_text_to_images(file_path)
            else:
                process_image_to_images(file_path)
        except Exception as rebuild_err:
            print(f"[Warning] Failed to rebuild cache for {doc_name}: {rebuild_err}")


@router.get("/files/{document_id}/download")
def download_pdf(document_id: str):
    """
    Download the original file (PDF or image).
    """
    pdfs_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
    matches = glob.glob(os.path.join(pdfs_dir, f"{document_id}_*"))
    
    file_path = matches[0] if matches else None
        
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
        
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {".pdf": "application/pdf", ".png": "image/png",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_map.get(ext, "application/octet-stream")
    
    return FileResponse(
        file_path, 
        media_type=mime, 
        content_disposition_type="inline", 
        filename=filename
    )

@router.get("/files")
def list_files():
    """
    Returns a list of all uniquely uploaded documents in the vector store.
    """
    from backend.main import vector_store_instance
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant backend is not ready")

    try:
        files = vector_store_instance.get_all_documents()
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{document_id}")
def delete_file(document_id: str):
    """
    Deletes a document from the vector store by its ID.
    """
    from backend.main import vector_store_instance
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant backend is not ready")
        
    try:
        vector_store_instance.delete_document(document_id)
        
        pdfs_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
        for m in glob.glob(os.path.join(pdfs_dir, f"{document_id}_*")):
            os.remove(m)
            
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md"}

@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """
    接收 PDF 或图片文件，处理为页面缓存与向量特征。
    支持格式：PDF / PNG / JPG / JPEG / WEBP
    """
    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="请上传支持的文件格式（PDF / PNG / JPG / WEBP / TXT / MD）")
    tmp = tempfile.NamedTemporaryFile(prefix="rag_upload_", suffix=suffix, delete=False)
    temp_path = tmp.name
    tmp.close()
    try:
        # 将流保存到本地临时文件
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. 解析文件到图像缓存
        if suffix == ".pdf":
            image_paths = process_pdf_to_images(temp_path)
        elif suffix in {".txt", ".md"}:
            image_paths = process_text_to_images(temp_path)
        else:
            image_paths = process_image_to_images(temp_path)
        
        if not image_paths:
            raise HTTPException(status_code=500, detail="未能成功解析出任何页面图像")

        file_hash = get_file_hash(temp_path)
        safe_filename = file.filename.replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        # 保存持久化 PDF
        pdfs_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
        os.makedirs(pdfs_dir, exist_ok=True)
        final_pdf_path = os.path.join(pdfs_dir, f"{file_hash}_{safe_filename}")
        shutil.copy(temp_path, final_pdf_path)
        
        # 避免并发时引错库，延迟导入或从全局拿， 这里我们可以从全局获取 vector_store_instance，或者重新调单例
        # 为简单起见，从 main 导入全局实例
        from backend.main import vector_store_instance
        
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
def chat(req: ChatRequest):
    """
    接收对某几篇（或全部）文档的查询，通过 SSE 流式返回：
    第一个事件为 evidence 卡片数据，后续事件为逐 token 文字，最终发送 [DONE]。
    """
    from backend.main import vector_store_instance
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")

    # ── 同步完成 RAG 检索（流式响应开始前必须就绪）──
    results = vector_store_instance.retrieve_with_two_stage(
        query_text=req.query,
        document_ids=req.document_ids,
        top_k=req.top_k
    )

    _rebuild_image_cache_if_needed(results)
    score_filtered = [r for r in results if float(r.get("score", 0)) >= req.min_score]
    if not score_filtered and results:
        score_filtered = results[:1]
    valid_results = [r for r in score_filtered if os.path.exists(str(r.get("image_path", "")))]
    evidence_images = [r["image_path"] for r in valid_results if r.get("image_path")]

    used_keys = {(r.get("document_id", ""), r.get("page_number", -1)) for r in valid_results}
    frontend_evidences = []
    for r in valid_results:
        image_path = str(r.get("image_path", ""))
        if not image_path:
            continue
        frontend_evidences.append({
            "document_name": r.get("document_name", "Unknown File"),
            "page_number": r.get("page_number", 0),
            "score": float(r.get("score", 0.0)),
            "image_base64": image_path_to_base64(image_path)
        })

    # 全部候选页（top_k 个，含未被 min_score 采用的），供前端展示所有得分
    all_candidates_fe = []
    for r in results:
        image_path = str(r.get("image_path", ""))
        if not image_path:
            continue
        key = (r.get("document_id", ""), r.get("page_number", -1))
        all_candidates_fe.append({
            "document_name": r.get("document_name", "Unknown File"),
            "page_number": r.get("page_number", 0),
            "score": float(r.get("score", 0.0)),
            "image_base64": image_path_to_base64(image_path),
            "is_used": key in used_keys,
        })

    def event_stream():
        # ── 事件 1: 证据卡片 ──
        if not evidence_images:
            yield f"data: {json.dumps({'type': 'error', 'data': '抱歉，未能检索出与该问题高度匹配的页面。'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield f"data: {json.dumps({'type': 'evidences', 'data': {'evidences': frontend_evidences, 'all_candidates': all_candidates_fe}})}\n\n"

        # ── 事件 2..N: 逐 token 文字 ──
        for token in generate_answer_stream(
            query_text=req.query,
            image_paths=evidence_images,
            chat_history=req.chat_history,
        ):
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        # ── 结束哨兵 ──
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

def image_path_to_base64(image_path: str) -> str:
    import base64
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    mime = "image/png"
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        mime = "image/jpeg"
    return f"data:{mime};base64,{encoded}"