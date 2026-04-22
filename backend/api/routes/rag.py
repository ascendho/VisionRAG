from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import tempfile
import glob

from src.pdf_processor import process_pdf_to_images, get_file_hash
from src.llm_generator import generate_answer_with_vision
from src.evidence_localizer import extract_and_cache_page_regions, build_localized_evidences

router = APIRouter()

# DTO schema for chat request
class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    chat_history: Optional[List[dict]] = None
    top_k: int = 3

@router.get("/files/{document_id}/download")
def download_pdf(document_id: str):
    """
    Download the original PDF file.
    """
    pdfs_dir = os.path.join(os.getcwd(), "qdrant_local", "pdfs")
    pattern = os.path.join(pdfs_dir, f"{document_id}_*.pdf")
    matches = glob.glob(pattern)
    
    fallback_path = os.path.join(pdfs_dir, f"{document_id}.pdf")
    
    pdf_path = None
    if matches:
        pdf_path = matches[0]
    elif os.path.exists(fallback_path):
        pdf_path = fallback_path
        
    if not pdf_path:
        raise HTTPException(status_code=404, detail="File not found")
        
    filename = os.path.basename(pdf_path)
    
    return FileResponse(
        pdf_path, 
        media_type="application/pdf", 
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
        pattern = os.path.join(pdfs_dir, f"{document_id}_*.pdf")
        for m in glob.glob(pattern):
            os.remove(m)
            
        fallback_path = os.path.join(pdfs_dir, f"{document_id}.pdf")
        if os.path.exists(fallback_path):
            os.remove(fallback_path)
            
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
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

        try:
            extract_and_cache_page_regions(temp_path, image_paths)
        except Exception as region_err:
            print(f"[Warning] 页面区域提取失败，将回退到整页证据: {region_err}")
            
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
    接收对某几篇（或全部）文档的查询，
    进行两阶段召回后使用豆包生成答案，返回文字回答以及匹配到的页面图片（供溯源）。
    """
    try:
        from backend.main import vector_store_instance
        if vector_store_instance is None:
            raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")
            
        # 1. RAG 图像双路检索
        results = vector_store_instance.retrieve_with_two_stage(
            query_text=req.query,
            document_ids=req.document_ids,
            top_k=req.top_k
        )
        
        localized_evidences = build_localized_evidences(req.query, results)
        evidence_images = [evidence["image_path"] for evidence in localized_evidences if evidence.get("image_path")]

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

        frontend_evidences = []
        for evidence in localized_evidences:
            image_path = evidence.get("image_path", "")
            if not image_path:
                continue
            frontend_evidences.append({
                "document_name": evidence.get("document_name", "Unknown File"),
                "page_number": evidence.get("page_number", 0),
                "score": float(evidence.get("score", 0.0)),
                "image_kind": evidence.get("image_kind", "page"),
                "image_size": evidence.get("image_size", []),
                "regions": evidence.get("regions", []),
                "image_base64": image_path_to_base64(image_path)
            })
        
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