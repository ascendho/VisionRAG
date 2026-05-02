from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging
import os
import json
import re
import shutil
import tempfile
import glob
import time

from src.doc_processor import process_pdf_to_images, process_image_to_images, process_text_to_images, get_file_hash
from src.llm_generator import generate_answer_stream, generate_suggested_questions
from src.config import DEFAULT_MIN_SCORE, QUERY_GUARD_ENABLED

router = APIRouter()
logger = logging.getLogger(__name__)

# DTO schema for chat request
class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    chat_history: Optional[List[dict]] = None
    top_k: int = 5
    min_score: float = DEFAULT_MIN_SCORE


_PRE_RETRIEVAL_GUARD_PATTERNS = [
    (
        "assistant_identity",
        re.compile(r"^(你是谁|你是誰|你叫什么|你叫啥|你叫什麼|介绍你自己|介绍一下你自己|请介绍一下你自己|你能做什么|你会什么)$", re.IGNORECASE),
    ),
    (
        "small_talk",
        re.compile(r"^(你好|您好|hello|hi|hey)$", re.IGNORECASE),
    ),
    (
        "off_topic_smalltalk",
        re.compile(r"^(天气怎么样|今天天气怎么样|讲个笑话|说个笑话)$", re.IGNORECASE),
    ),
]

_COMPOUND_QUERY_SPLIT_PATTERN = re.compile(r"[？?；;\n]+")


def _normalize_query_text(query: str) -> str:
    return re.sub(r"[\s\.,!?，。！？、:：;；\"'“”‘’()（）【】\[\]<>《》]+", "", query.lower()).strip()


def _build_guarded_payload(reason: str, stage: str, message: Optional[str] = None, meta: Optional[dict] = None) -> dict:
    if message is None:
        if reason == "assistant_identity":
            message = (
                "这个问题不属于当前文档问答范围。"
                "我只能依据已上传文档中的证据回答问题，暂时不能回答关于助手身份或能力设定的问题。"
                "请改问与文档内容直接相关的问题。"
            )
        else:
            message = (
                "这个问题与当前文档内容范围无关，或缺少足够可靠的文档证据。"
                "我暂时不返回依据页，也不会基于外部常识作答。"
                "请换一个更贴近文档内容的问题。"
            )

    return {
        "message": message,
        "reason": reason,
        "stage": stage,
        "meta": meta or {},
    }


def _guard_obvious_out_of_scope(query: str) -> Optional[dict]:
    if not QUERY_GUARD_ENABLED:
        return None

    normalized = _normalize_query_text(query)
    if not normalized:
        return None

    for reason, pattern in _PRE_RETRIEVAL_GUARD_PATTERNS:
        if pattern.match(normalized):
            return _build_guarded_payload(reason=reason, stage="pre_retrieval")

    return None


def _split_compound_query(query: str) -> List[str]:
    normalized_query = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized_query:
        return []

    raw_parts = [
        part.strip(" \t\r\n?？;；")
        for part in _COMPOUND_QUERY_SPLIT_PATTERN.split(normalized_query)
    ]

    deduped_parts: List[str] = []
    seen_parts = set()
    for part in raw_parts:
        if not part:
            continue
        normalized_part = _normalize_query_text(part)
        if not normalized_part or normalized_part in seen_parts:
            continue
        seen_parts.add(normalized_part)
        deduped_parts.append(part)

    return deduped_parts or [normalized_query]


def _result_identity(result: Dict[str, Any]) -> tuple:
    return (
        result.get("document_id", ""),
        result.get("page_number", -1),
        result.get("image_path", ""),
    )


def _filter_results_with_fallback(results: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    filtered = [dict(item) for item in results if float(item.get("score", 0.0)) >= min_score]
    if not filtered and results:
        filtered = [dict(results[0])]
    return filtered


def _merge_result_groups(
    result_groups: List[List[Dict[str, Any]]],
    sub_queries: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    merged_by_key: Dict[tuple, Dict[str, Any]] = {}
    merged_order: List[Dict[str, Any]] = []
    max_group_len = max((len(group) for group in result_groups), default=0)

    for rank in range(max_group_len):
        for part_index, group in enumerate(result_groups):
            if rank >= len(group):
                continue

            candidate = dict(group[rank])
            key = _result_identity(candidate)
            if key in merged_by_key:
                existing = merged_by_key[key]
                existing["score"] = max(float(existing.get("score", 0.0)), float(candidate.get("score", 0.0)))
                matched = existing.setdefault("matched_sub_queries", [])
                sub_query = sub_queries[part_index]
                if sub_query not in matched:
                    matched.append(sub_query)
                continue

            candidate["matched_sub_queries"] = [sub_queries[part_index]]
            merged_by_key[key] = candidate
            merged_order.append(candidate)

    return merged_order[:limit]


def _aggregate_compound_retrieval_timing(
    sub_queries: List[str],
    timings: List[Dict[str, Any]],
    returned_points: int,
) -> Dict[str, Any]:
    return {
        "query_embedding_ms": round(sum(float(item.get("query_embedding_ms", 0.0)) for item in timings), 2),
        "qdrant_query_ms": round(sum(float(item.get("qdrant_query_ms", 0.0)) for item in timings), 2),
        "result_format_ms": round(sum(float(item.get("result_format_ms", 0.0)) for item in timings), 2),
        "total_retrieval_ms": round(sum(float(item.get("total_retrieval_ms", 0.0)) for item in timings), 2),
        "prefetch_limit": sum(int(item.get("prefetch_limit", 0)) for item in timings),
        "returned_points": returned_points,
        "sub_query_count": len(sub_queries),
        "sub_query_timings": [
            {
                "query": sub_query,
                "query_embedding_ms": timing.get("query_embedding_ms", 0.0),
                "qdrant_query_ms": timing.get("qdrant_query_ms", 0.0),
                "result_format_ms": timing.get("result_format_ms", 0.0),
                "total_retrieval_ms": timing.get("total_retrieval_ms", 0.0),
                "returned_points": timing.get("returned_points", 0),
            }
            for sub_query, timing in zip(sub_queries, timings)
        ],
    }


def _retrieve_compound_aware(
    vector_store_instance,
    query_text: str,
    document_ids: Optional[List[str]],
    top_k: int,
    min_score: float,
) -> Dict[str, Any]:
    sub_queries = _split_compound_query(query_text)
    if len(sub_queries) <= 1:
        retrieval_payload = vector_store_instance.retrieve_with_two_stage(
            query_text=query_text,
            document_ids=document_ids,
            top_k=top_k,
        )
        results = retrieval_payload["results"]
        return {
            "sub_queries": sub_queries or [query_text],
            "results": results,
            "selected_results": _filter_results_with_fallback(results, min_score),
            "timing": retrieval_payload["timing"],
        }

    per_sub_query_top_k = max(2, min(3, top_k))
    selected_limit = min(8, max(top_k, len(sub_queries) * per_sub_query_top_k))
    candidate_limit = min(12, max(selected_limit, len(sub_queries) * per_sub_query_top_k))

    raw_result_groups: List[List[Dict[str, Any]]] = []
    selected_result_groups: List[List[Dict[str, Any]]] = []
    sub_query_timings: List[Dict[str, Any]] = []

    for sub_query in sub_queries:
        retrieval_payload = vector_store_instance.retrieve_with_two_stage(
            query_text=sub_query,
            document_ids=document_ids,
            top_k=per_sub_query_top_k,
        )
        raw_results = retrieval_payload["results"]
        raw_result_groups.append(raw_results)
        selected_result_groups.append(_filter_results_with_fallback(raw_results, min_score))
        sub_query_timings.append(retrieval_payload["timing"])

    merged_results = _merge_result_groups(raw_result_groups, sub_queries, candidate_limit)
    merged_selected_results = _merge_result_groups(selected_result_groups, sub_queries, selected_limit)

    return {
        "sub_queries": sub_queries,
        "results": merged_results,
        "selected_results": merged_selected_results,
        "timing": _aggregate_compound_retrieval_timing(sub_queries, sub_query_timings, len(merged_results)),
    }


def _build_confidence_summary(results: List[dict]) -> dict:
    scores = sorted((float(item.get("score", 0.0)) for item in results), reverse=True)[:3]
    if not scores:
        return {"label": "低", "score": 0.0, "sample_size": 0, "tier": "low"}

    avg_score = sum(scores) / len(scores)
    if avg_score >= 0.85:
        label = "高"
        tier = "high"
    elif avg_score >= 0.7:
        label = "中"
        tier = "medium"
    else:
        label = "低"
        tier = "low"

    return {
        "label": label,
        "score": round(avg_score, 2),
        "sample_size": len(scores),
        "tier": tier,
    }


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
        image_render_start = time.perf_counter()
        if suffix == ".pdf":
            image_paths = process_pdf_to_images(temp_path)
        elif suffix in {".txt", ".md"}:
            image_paths = process_text_to_images(temp_path)
        else:
            image_paths = process_image_to_images(temp_path)
        document_render_ms = (time.perf_counter() - image_render_start) * 1000
        
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
        index_result = vector_store_instance.embed_and_store_documents(
            image_paths=image_paths,
            document_id=file_hash,
            document_name=file.filename
        )

        if not index_result.get("ok"):
            raise HTTPException(status_code=500, detail="存储多模态特征库发生意外错误")

        upload_timing = {
            "document_render_ms": round(document_render_ms, 2),
            **index_result.get("timing", {}),
        }
        logger.info("upload_timing file=%s timing=%s", file.filename, upload_timing)
        print(
            "[Upload] "
            f"file={file.filename} render={upload_timing['document_render_ms']}ms "
            f"embed={upload_timing.get('embedding_ms', 0)}ms build={upload_timing.get('point_build_ms', 0)}ms "
            f"upsert={upload_timing.get('qdrant_upsert_ms', 0)}ms total_index={upload_timing.get('total_index_ms', 0)}ms"
        )

        return {
            "status": "success",
            "document_id": file_hash,
            "document_name": file.filename,
            "page_count": len(image_paths),
            "timing": upload_timing,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 无论成功错误，清除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/files/{document_id}/suggestions")
def get_document_suggestions(document_id: str, max_questions: int = 4):
    from backend.main import vector_store_instance

    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")

    sample_pages = vector_store_instance.get_document_page_samples(
        document_id=document_id,
        limit=max(3, min(max_questions, 5)),
    )
    if not sample_pages:
        raise HTTPException(status_code=404, detail="未找到可用于生成建议问题的文档页面")

    _rebuild_image_cache_if_needed(sample_pages)
    available_pages = [page for page in sample_pages if os.path.exists(str(page.get("image_path", "")))]
    if not available_pages:
        raise HTTPException(status_code=404, detail="建议问题所需的页面缓存不可用")

    evidence_context = []
    image_paths = []
    for index, page in enumerate(available_pages, start=1):
        image_paths.append(page["image_path"])
        evidence_context.append(
            {
                "evidence_id": f"E{index}",
                "document_name": page.get("document_name", "Unknown File"),
                "page_number": page.get("page_number", 0),
            }
        )

    questions = generate_suggested_questions(
        document_name=available_pages[0].get("document_name", document_id),
        image_paths=image_paths,
        evidence_context=evidence_context,
        max_questions=max_questions,
    )

    return {
        "status": "success",
        "document_id": document_id,
        "document_name": available_pages[0].get("document_name", document_id),
        "questions": questions,
    }

@router.post("/chat")
async def chat(request: Request, req: ChatRequest):
    """
    接收对某几篇（或全部）文档的查询，通过 SSE 流式返回：
    第一个事件为 evidence 卡片数据，后续事件为逐 token 文字，最终发送 [DONE]。
    """
    from backend.main import vector_store_instance
    if vector_store_instance is None:
        raise HTTPException(status_code=500, detail="Qdrant 后端尚未就绪")

    pre_guard = _guard_obvious_out_of_scope(req.query)

    results = []
    score_filtered: List[Dict[str, Any]] = []
    retrieval_timing = None
    sub_queries = [req.query]

    # ── 同步完成 RAG 检索（流式响应开始前必须就绪）──
    if not pre_guard:
        retrieval_payload = _retrieve_compound_aware(
            vector_store_instance=vector_store_instance,
            query_text=req.query,
            document_ids=req.document_ids,
            top_k=req.top_k,
            min_score=req.min_score,
        )
        results = retrieval_payload["results"]
        score_filtered = retrieval_payload["selected_results"]
        retrieval_timing = retrieval_payload["timing"]
        sub_queries = retrieval_payload["sub_queries"]

    _rebuild_image_cache_if_needed(results)
    valid_results = [r for r in score_filtered if os.path.exists(str(r.get("image_path", "")))]
    evidence_images = [r["image_path"] for r in valid_results if r.get("image_path")]
    confidence = _build_confidence_summary(valid_results)

    used_keys = {(r.get("document_id", ""), r.get("page_number", -1)) for r in valid_results}
    evidence_id_map = {}
    frontend_evidences = []
    evidence_context = []
    for index, r in enumerate(valid_results, start=1):
        image_path = str(r.get("image_path", ""))
        if not image_path:
            continue
        key = (r.get("document_id", ""), r.get("page_number", -1))
        evidence_id = f"E{index}"
        evidence_id_map[key] = evidence_id
        evidence_context.append({
            "evidence_id": evidence_id,
            "document_name": r.get("document_name", "Unknown File"),
            "page_number": r.get("page_number", 0),
            "score": float(r.get("score", 0.0)),
            "matched_sub_queries": list(r.get("matched_sub_queries", [])),
        })
        frontend_evidences.append({
            "evidence_id": evidence_id,
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
            "evidence_id": evidence_id_map.get(key),
            "document_name": r.get("document_name", "Unknown File"),
            "page_number": r.get("page_number", 0),
            "score": float(r.get("score", 0.0)),
            "image_base64": image_path_to_base64(image_path),
            "is_used": key in used_keys,
        })

    if pre_guard:
        guard_payload = pre_guard
        logger.info(
            "chat_guarded stage=%s reason=%s query=%r document_scope=%s meta=%s",
            guard_payload["stage"],
            guard_payload["reason"],
            req.query,
            req.document_ids,
            guard_payload.get("meta", {}),
        )
        print(
            f"[Chat Guard] stage={guard_payload['stage']} reason={guard_payload['reason']} "
            f"document_scope={req.document_ids} meta={guard_payload.get('meta', {})}"
        )
    else:
        logger.info(
            "chat_retrieval timing=%s confidence=%s document_scope=%s sub_queries=%s",
            retrieval_timing,
            confidence,
            req.document_ids,
            sub_queries,
        )
        print(
            f"[Chat] confidence={confidence} timing={retrieval_timing} "
            f"document_scope={req.document_ids} sub_queries={sub_queries}"
        )

    async def event_stream():
        if pre_guard:
            guard_payload = pre_guard
            yield f"data: {json.dumps({'type': 'guarded', 'data': guard_payload})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── 事件 1: 证据卡片 ──
        if await request.is_disconnected():
            return

        if not evidence_images:
            yield f"data: {json.dumps({'type': 'error', 'data': '抱歉，未能检索出与该问题高度匹配的页面。'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield f"data: {json.dumps({'type': 'evidences', 'data': {'evidences': frontend_evidences, 'all_candidates': all_candidates_fe, 'confidence': confidence, 'retrieval_timing': retrieval_timing}})}\n\n"

        # ── 事件 2..N: 逐 token 文字 ──
        token_stream = generate_answer_stream(
            query_text=req.query,
            image_paths=evidence_images,
            chat_history=req.chat_history,
            evidence_context=evidence_context,
            sub_queries=sub_queries,
        )
        generation_start = time.perf_counter()
        first_token_ms = None
        token_count = 0
        try:
            for token in token_stream:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - generation_start) * 1000
                token_count += 1
                if await request.is_disconnected():
                    return
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        finally:
            total_generation_ms = (time.perf_counter() - generation_start) * 1000
            generation_timing = {
                "first_token_ms": round(first_token_ms or 0.0, 2),
                "total_generation_ms": round(total_generation_ms, 2),
                "token_count": token_count,
            }
            logger.info("chat_generation timing=%s document_scope=%s", generation_timing, req.document_ids)
            print(
                "[Generation] "
                f"first_token={generation_timing['first_token_ms']}ms "
                f"total={generation_timing['total_generation_ms']}ms tokens={token_count} "
                f"document_scope={req.document_ids}"
            )
            token_stream.close()

        # ── 结束哨兵 ──
        if await request.is_disconnected():
            return
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