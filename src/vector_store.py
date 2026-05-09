"""视觉向量存储与检索模块。

这个文件实现了项目里最核心的“检索侧”能力：
1. 把文档页图片送入 ColPali，得到可用于 Late Interaction 的多向量表示。
2. 再把原始多向量压缩成 MUVERA 表示，用于第一阶段快速召回。
3. 把两套向量一起写入 Qdrant，并在查询时执行“两阶段检索”：
    先用压缩向量粗召回，再用原始多向量精排。

如果把整个项目看成一条流水线，那么：
- `doc_processor.py` 负责把输入变成页面图；
- 这里负责把页面图变成“可检索的向量索引”；
- `llm_generator.py` 则负责把检索出的证据页交给大模型生成答案。
"""

import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional
import warnings

import numpy as np

# 屏蔽一类已知但通常无害的 PyTorch 警告，避免本地终端被噪声刷屏。
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

from qdrant_client import QdrantClient, models
from fastembed.postprocess.muvera import Muvera
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image

from src.config import (
    COLLECTION_NAME,
    COLPALI_MODEL_NAME,
    DEFAULT_RETRIEVAL_MODE,
    QDRANT_PATH,
    QDRANT_URL,
    SUPPORTED_RETRIEVAL_MODES,
)

logger = logging.getLogger(__name__)

COLPALI_VECTOR_DIM = 128
MUVERA_K_SIM = 6
MUVERA_DIM_PROJ = 16
MUVERA_R_REPS = 30
MUVERA_RANDOM_SEED = 42
MUVERA_SCORE_SCALE = 2 ** (MUVERA_K_SIM / 2)


def get_muvera_fde_dimension(
    *,
    k_sim: int = MUVERA_K_SIM,
    dim_proj: int = MUVERA_DIM_PROJ,
    r_reps: int = MUVERA_R_REPS,
) -> int:
    return int(r_reps * (2 ** k_sim) * dim_proj)


class VisionVectorStore:
    """
    视觉向量数据库服务。

    这个类把“建索引”和“查索引”两件事都包了起来：
    1. 建索引时，把每一页文档图片编码成 ColPali 原始多向量，并额外生成 MUVERA 压缩向量。
    2. 查索引时，先用 MUVERA 压缩向量做第一阶段粗召回，再用 ColPali 原始多向量精排。
    3. 最终把结果组织成上层编排代码更容易消费的字典结构。

    之所以不直接只存一套原始向量，是因为原始多向量虽然精度高，但全量扫描成本也高；
    增加一套压缩表示后，可以先快速缩小候选范围，再对少量候选做高精度比较。
    """

    def __init__(self):
        # 连接本地 Qdrant。`check_compatibility=False` 是为了忽略次版本差异带来的提示，
        # 避免开发环境中客户端与服务端小版本不完全一致时频繁报警。
        self.qdrant = self._build_qdrant_client()
        
        # MUVERA 负责把 ColPali 的原始多向量压成固定维度单向量，主要服务第一阶段粗召回。
        self.muvera = Muvera(
            dim=COLPALI_VECTOR_DIM,      # ColPali 单个向量 token 的原始维度。
            k_sim=MUVERA_K_SIM,          # 聚类数量控制参数，对应 2^6 = 64 个聚类中心。
            dim_proj=MUVERA_DIM_PROJ,    # 每个聚类中心随机投影后的维度。
            r_reps=MUVERA_R_REPS,        # 重复投影次数越高，近似召回通常越稳定，但计算也会略增。
            random_seed=MUVERA_RANDOM_SEED,
        )
        
        # 按硬件能力选择运行设备：优先 CUDA，其次 Apple MPS，最后退回 CPU。
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # processor 负责把图片或文本查询整理成模型可接受的张量输入。
        self.processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_NAME)
        # 使用 bfloat16 的主要目的是降低本地显存/内存压力，减少加载或推理时 OOM 的风险。
        dtype = torch.bfloat16
        
        # `low_cpu_mem_usage=True` 可以降低模型加载时的峰值内存压力。
        self.model = ColPali.from_pretrained(
            COLPALI_MODEL_NAME,
            torch_dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        
        # 启动时确保业务集合存在，这样上传与查询可以直接复用同一套结构定义。
        self._ensure_collection_exists()

    def _build_qdrant_client(self) -> QdrantClient:
        if QDRANT_URL:
            try:
                client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
                client.get_collections()
                return client
            except Exception as exc:
                logger.warning(
                    "Failed to connect to remote Qdrant at %s, falling back to local path %s: %s",
                    QDRANT_URL,
                    QDRANT_PATH,
                    exc,
                )

        logger.info("Using embedded Qdrant storage at %s", QDRANT_PATH)
        return QdrantClient(path=QDRANT_PATH)

    def _ensure_collection_exists(self):
        """
        初始化 Qdrant 集合结构。

        这里最关键的设计是：同一页文档会被写入两套向量池。
        1. `original`: ColPali 原始多向量，精度高，用于第二阶段精排。
        2. `muvera_fde`: MUVERA 压缩向量，体积更小，用于第一阶段粗召回。

        只有把这两套向量都预先存好，查询时才能在一次检索流程里完成
        “先快后准”的两阶段策略。
        """
        if not self.qdrant.collection_exists(COLLECTION_NAME):
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=self._build_vectors_config(),
            )
            return

        self._validate_collection_schema()

    @property
    def muvera_fde_dimension(self) -> int:
        return int(getattr(self.muvera, "embedding_size", get_muvera_fde_dimension()))

    def _build_vectors_config(self) -> Dict[str, models.VectorParams]:
        return {
            "original": models.VectorParams(
                size=COLPALI_VECTOR_DIM,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            "muvera_fde": models.VectorParams(
                size=self.muvera_fde_dimension,
                distance=models.Distance.DOT,
            ),
        }

    def _get_vector_params(self, collection_info: Any, vector_name: str) -> Any:
        config = getattr(collection_info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if isinstance(vectors, dict):
            return vectors.get(vector_name)
        return None

    def _get_vector_param_value(self, vector_params: Any, field_name: str) -> Any:
        if vector_params is None:
            return None
        if isinstance(vector_params, dict):
            return vector_params.get(field_name)
        return getattr(vector_params, field_name, None)

    def _validate_collection_schema(self) -> None:
        collection_info = self.qdrant.get_collection(COLLECTION_NAME)
        original_params = self._get_vector_params(collection_info, "original")
        muvera_params = self._get_vector_params(collection_info, "muvera_fde")

        issues = []
        original_size = self._get_vector_param_value(original_params, "size")
        original_multivector = self._get_vector_param_value(original_params, "multivector_config")
        if original_params is None:
            issues.append("missing vector field 'original'")
        elif int(original_size or 0) != COLPALI_VECTOR_DIM or original_multivector is None:
            issues.append("vector field 'original' must be a 128-D multivector")

        muvera_size = self._get_vector_param_value(muvera_params, "size")
        muvera_multivector = self._get_vector_param_value(muvera_params, "multivector_config")
        if muvera_params is None:
            issues.append("missing vector field 'muvera_fde'")
        elif int(muvera_size or 0) != self.muvera_fde_dimension or muvera_multivector is not None:
            issues.append(
                "vector field 'muvera_fde' must be a "
                f"{self.muvera_fde_dimension}-D single dense vector"
            )

        if issues:
            raise RuntimeError(
                f"Qdrant collection {COLLECTION_NAME!r} has an incompatible vector schema: "
                + "; ".join(issues)
                + ". Reset and reindex the collection before querying with the MUVERA FDE pipeline."
            )

    def reset_collection(self):
        """
        完全清空并重建当前的 Qdrant 集合。
        这通常用于显式重置索引状态，而不是日常查询路径的一部分。
        """
        if self.qdrant.collection_exists(COLLECTION_NAME):
            self.qdrant.delete_collection(COLLECTION_NAME)
        self._ensure_collection_exists()

    def _make_point_id(self, document_id: str, page_number: int) -> int:
        """
        为“某个文档的某一页”生成稳定 ID。

        使用稳定 ID 有两个直接好处：
        1. 同一文档重复上传时，可以更可靠地覆盖旧数据而不是无限叠加。
        2. 问题排查时，更容易把数据库中的点和具体页面对应起来。

        这里额外裁成 63 bit，是为了保持在常见有符号整数范围内。
        """
        digest = hashlib.sha1(f"{document_id}:{page_number}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) & ((1 << 63) - 1)

    def _build_document_filter(self, document_ids: Optional[List[str]]) -> Optional[models.Filter]:
        """
        根据文档 ID 列表构建 Qdrant 过滤器。

        上层如果传入 document_ids，就只在指定文档范围内检索；
        否则默认在整库中搜索。
        """
        conditions: List[Any] = [self._build_ready_document_condition()]

        if not document_ids:
            return models.Filter(must=conditions)

        clean_ids = sorted({doc_id for doc_id in document_ids if doc_id})
        if not clean_ids:
            return models.Filter(must=conditions)

        if len(clean_ids) == 1:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=clean_ids[0]),
                )
            )
            return models.Filter(must=conditions)

        conditions.append(
            models.Filter(
                should=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=doc_id),
                    )
                    for doc_id in clean_ids
                ]
            )
        )

        return models.Filter(must=conditions)

    def _build_ready_document_condition(self) -> models.Filter:
        return models.Filter(
            should=[
                models.FieldCondition(
                    key="index_ready",
                    match=models.MatchValue(value=True),
                ),
                models.IsNullCondition(
                    is_null=models.PayloadField(key="index_ready"),
                ),
            ]
        )

    def delete_document(self, document_id: str):
        """
        按 document_id 删除旧页面向量。

        这样做的主要目的是让同一文档重新上传时，不会把旧页和新页混在一起。
        """
        if not document_id:
            return

        self.qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def embed_and_store_documents(
        self,
        image_paths: List[str],
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
        batch_size: int = 4,
        replace_document: bool = True,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        **_ignored,
    ) -> Dict[str, Any]:
        """
        为一组页面图片建立索引并写入 Qdrant。

        这个函数做了四件连续的事情：
        1. 批量读取页面图片并送入 ColPali，得到原始多向量表示。
        2. 对每一页的原始多向量再计算一份 MUVERA 压缩向量。
        3. 把两套向量和页面元数据组装成 Qdrant Point。
        4. 分批写入 Qdrant，避免单次请求体过大。

        整个过程按批处理，是因为页面多时模型推理和中间张量都很占内存。
        
        参数:
            image_paths (List[str]): 用于处理和存储的多页图片的物理路径。
            batch_size (int): 每次送入模型的并行图片数，默认为 4。
        """
        document_id = document_id or hashlib.sha1("\n".join(image_paths).encode("utf-8")).hexdigest()
        document_name = document_name or document_id
        total_index_start = time.perf_counter()
        embedding_ms = 0.0
        point_build_ms = 0.0

        # 同一 document_id 重新入库时，默认先清理旧页，避免历史脏数据残留。
        if replace_document:
            self.delete_document(document_id)

        total_images = len(image_paths)
        points = []
        
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            try:
                embedding_start = time.perf_counter()
                # 先把 PIL 图片转成 ColPali 所需的批量张量格式。
                batch_inputs = self.processor.process_images(images).to(self.device).to(self.model.dtype)

                with torch.no_grad():
                    # 输出结果不是单个向量，而是一页对应多个 token 向量的多向量表示。
                    image_embeddings = self.model(**batch_inputs)
                embedding_ms += (time.perf_counter() - embedding_start) * 1000
                if on_progress:
                    on_progress(
                        {
                            "stage": "embedding",
                            "completed": min(i + len(batch_paths), total_images),
                            "total": total_images,
                        }
                    )
            finally:
                # PIL 句柄及时关闭，避免大批量处理时文件句柄和内存积累。
                for img in images:
                    img.close()

            point_build_start = time.perf_counter()
            # 模型输出先转到 CPU/Numpy，便于后续交给 MUVERA 和 Qdrant 客户端处理。
            cpu_embeddings = [emb.float().cpu().numpy() for emb in image_embeddings]
            
            for j, (path, original_emb) in enumerate(zip(batch_paths, cpu_embeddings)):
                # 无论上游输出形状如何，入库前都显式重塑成“若干行 x 128 列”。
                # 这一步的目标是确保每一行都表示一个 token 向量，便于后续多向量检索。
                original_emb_2d = np.asarray(original_emb, dtype=np.float32).reshape(-1, COLPALI_VECTOR_DIM)
                muvera_fde = self._encode_muvera_document(original_emb_2d)
                
                # 页码从 1 开始，和用户看到的文档页概念保持一致。
                page_number = i + j + 1
                point = models.PointStruct(
                    id=self._make_point_id(document_id=document_id, page_number=page_number),
                    vector={
                        "original": original_emb_2d.tolist(),
                        "muvera_fde": muvera_fde.tolist(),
                    },
                    payload={
                        "image_path": path,
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_number,
                        "index_ready": False,
                    }
                )
                points.append(point)
            point_build_ms += (time.perf_counter() - point_build_start) * 1000
                
        # 没有页面就直接返回失败结果，避免后续 upsert 空数组。
        if not points:
            return {
                "ok": False,
                "timing": {
                    "pages": total_images,
                    "embedding_ms": round(embedding_ms, 2),
                    "point_build_ms": round(point_build_ms, 2),
                    "qdrant_upsert_ms": 0.0,
                    "total_index_ms": round((time.perf_counter() - total_index_start) * 1000, 2),
                    "device": self.device,
                    "batch_size": batch_size,
                },
            }

        # Qdrant 单次请求体过大时容易失败，所以这里继续把构建好的点分批提交。
        # 之所以选择 8 页一批，是基于当前向量 JSON 体积做的保守上限控制。
        upsert_batch_size = 8
        upsert_start = time.perf_counter()
        for i in range(0, len(points), upsert_batch_size):
            batch_points = points[i:i + upsert_batch_size]
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=batch_points
            )
            if on_progress:
                on_progress(
                    {
                        "stage": "upserting",
                        "completed": min(i + len(batch_points), len(points)),
                        "total": len(points),
                    }
                )

        self.qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"index_ready": True},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
        )
        qdrant_upsert_ms = (time.perf_counter() - upsert_start) * 1000
        total_index_ms = (time.perf_counter() - total_index_start) * 1000
        timing = {
            "pages": total_images,
            "embedding_ms": round(embedding_ms, 2),
            "point_build_ms": round(point_build_ms, 2),
            "qdrant_upsert_ms": round(qdrant_upsert_ms, 2),
            "total_index_ms": round(total_index_ms, 2),
            "device": self.device,
            "batch_size": batch_size,
            "upsert_batch_size": upsert_batch_size,
        }
        logger.info("index_timing document_id=%s timing=%s", document_id, timing)
        print(
            "[Index] "
            f"document_id={document_id} pages={total_images} device={self.device} "
            f"embed={timing['embedding_ms']}ms build={timing['point_build_ms']}ms "
            f"upsert={timing['qdrant_upsert_ms']}ms total={timing['total_index_ms']}ms"
        )
        return {"ok": True, "timing": timing}

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        获取当前向量库中所有已装载的独立文件列表。
        这里不是读取某张专门的“文档表”，而是通过滚动扫描页面点，再按 document_id 聚合。
        """
        try:
            records, _ = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=self._build_document_filter(None),
                with_payload=["document_id", "document_name", "page_number"],
                with_vectors=False,
                limit=10000
            )
            
            docs = {}
            for r in records:
                doc_id = r.payload.get("document_id")
                if doc_id:
                    if doc_id not in docs:
                        docs[doc_id] = {
                            "document_id": doc_id,
                            "document_name": r.payload.get("document_name", "Unknown File"),
                            "page_count": 0
                        }
                    docs[doc_id]["page_count"] += 1
                    
            return list(docs.values())
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []

    def get_document_page_samples(self, document_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        """返回指定文档的代表页，用于建议问题等轻量功能。"""
        if not document_id:
            return []

        try:
            records, _ = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        self._build_ready_document_condition(),
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                with_payload=["image_path", "document_id", "document_name", "page_number"],
                with_vectors=False,
                limit=max(limit * 4, 50),
            )
        except Exception as exc:
            logger.warning("failed_to_load_document_page_samples document_id=%s error=%s", document_id, exc)
            return []

        pages = []
        for record in records:
            payload = record.payload or {}
            image_path = payload.get("image_path", "")
            if not image_path:
                continue
            pages.append(
                {
                    "image_path": image_path,
                    "document_id": payload.get("document_id", ""),
                    "document_name": payload.get("document_name", "Unknown File"),
                    "page_number": payload.get("page_number", 0),
                }
            )

        pages.sort(key=lambda item: item.get("page_number", 0))
        return pages[:limit]

    def _encode_query_tokens(self, query_text: str) -> np.ndarray:
        """把单条查询编码成 ColPali 查询多向量。"""
        batch_queries = self.processor.process_queries([query_text]).to(self.device)
        with torch.no_grad():
            query_embedding_tensor = self.model(**batch_queries)[0]
        query_embeddings = query_embedding_tensor.float().cpu().numpy().reshape(-1, COLPALI_VECTOR_DIM)
        return np.asarray(query_embeddings, dtype=np.float32)

    def _load_original_vectors_for_page(self, document_id: str, page_number: int) -> Optional[np.ndarray]:
        """从 Qdrant 中反查某一页保存的原始多向量。"""
        try:
            records, _ = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        self._build_ready_document_condition(),
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                        models.FieldCondition(
                            key="page_number",
                            match=models.MatchValue(value=int(page_number)),
                        ),
                    ]
                ),
                with_payload=False,
                with_vectors=True,
                limit=1,
            )
        except Exception as exc:
            logger.warning(
                "failed_to_load_original_vectors document_id=%s page_number=%s error=%s",
                document_id,
                page_number,
                exc,
            )
            return None

        if not records:
            return None

        record = records[0]
        vector_payload = getattr(record, "vector", None)
        if isinstance(vector_payload, dict):
            original_vectors = vector_payload.get("original")
            if original_vectors is None and vector_payload:
                original_vectors = next(iter(vector_payload.values()))
        else:
            original_vectors = vector_payload

        if original_vectors is None:
            return None

        original_array = np.asarray(original_vectors, dtype=np.float32)
        if original_array.size == 0:
            return None
        if original_array.ndim == 1:
            original_array = original_array.reshape(-1, COLPALI_VECTOR_DIM)
        return original_array

    def _score_query_against_document_embeddings(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> float:
        """用和检索阶段一致的 MaxSim 近似打分方式评估单页支持度。"""
        if query_embeddings.size == 0 or document_embeddings.size == 0:
            return 0.0

        similarity_matrix = np.matmul(query_embeddings, document_embeddings.T)
        if similarity_matrix.size == 0:
            return 0.0

        token_scores = similarity_matrix.max(axis=1)
        n_query_tokens = max(int(query_embeddings.shape[0]), 1)
        return float(token_scores.sum()) / n_query_tokens

    def probe_query_support_for_results(
        self,
        query_text: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """只在已选 evidence 页上复核某个查询是否也能被这些页面支持。"""
        normalized_query = str(query_text or "").strip()
        if not normalized_query or not results:
            return []

        query_embeddings = self._encode_query_tokens(normalized_query)
        support_entries: List[Dict[str, Any]] = []

        for result in results:
            document_id = str(result.get("document_id", ""))
            page_number = int(result.get("page_number", 0) or 0)
            image_path = str(result.get("image_path", ""))
            if not document_id or page_number <= 0:
                continue

            document_embeddings = self._load_original_vectors_for_page(document_id=document_id, page_number=page_number)
            if document_embeddings is None:
                continue

            support_score = self._score_query_against_document_embeddings(
                query_embeddings=query_embeddings,
                document_embeddings=document_embeddings,
            )
            support_entries.append(
                {
                    "document_id": document_id,
                    "page_number": page_number,
                    "image_path": image_path,
                    "score": round(support_score, 4),
                }
            )

        return support_entries

    def _encode_colpali_query(self, query_text: str) -> np.ndarray:
        batch_queries = self.processor.process_queries([query_text]).to(self.device)
        with torch.no_grad():
            query_embedding_tensor = self.model(**batch_queries)[0]
        return query_embedding_tensor.float().cpu().numpy().reshape(-1, COLPALI_VECTOR_DIM)

    def _validate_muvera_fde(self, fde: np.ndarray, vector_kind: str) -> np.ndarray:
        fde_1d = np.asarray(fde, dtype=np.float32).reshape(-1)
        if fde_1d.shape[0] != self.muvera_fde_dimension:
            raise ValueError(
                f"MUVERA {vector_kind} FDE has dimension {fde_1d.shape[0]}, "
                f"expected {self.muvera_fde_dimension}."
            )
        return fde_1d

    def _encode_muvera_document(self, document_colpali_2d: np.ndarray) -> np.ndarray:
        return self._validate_muvera_fde(
            self.muvera.process_document(document_colpali_2d),
            "document",
        )

    def _encode_muvera_query(self, query_colpali_2d: np.ndarray) -> np.ndarray:
        return self._validate_muvera_fde(
            self.muvera.process_query(query_colpali_2d),
            "query",
        )

    def _format_query_results(
        self,
        points: List[Any],
        *,
        top_k: int,
        score_normalizer: float,
    ) -> List[Dict[str, Any]]:
        unique = []
        seen = set()
        normalized_score_divisor = max(float(score_normalizer), 1.0)
        for point in points:
            payload = point.payload or {}
            key = (
                payload.get("document_id", ""),
                payload.get("page_number", -1),
                payload.get("image_path", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(
                {
                    "image_path": payload.get("image_path", ""),
                    "score": point.score / normalized_score_divisor,
                    "document_id": payload.get("document_id", ""),
                    "document_name": payload.get("document_name", ""),
                    "page_number": payload.get("page_number", 0),
                }
            )
            if len(unique) >= top_k:
                break

        return unique

    def retrieve_with_two_stage(
        self,
        query_text: str,
        top_k: int = 3,
        prefetch_multiplier: int = 10,
        document_ids: Optional[List[str]] = None,
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    ) -> Dict[str, Any]:
        """
        执行两阶段检索。

        查询阶段的核心策略是：
        1. 先把用户问题编码成 ColPali 查询多向量，再额外生成一份 MUVERA 压缩表示。
        2. 第一阶段用 `muvera_fde` 在全库中快速召回较大的候选集。
        3. 第二阶段只在候选集上用 `original` 原始多向量做精排。

        这样既能保住较高检索质量，又避免每次都拿原始多向量去扫全库。
        
        参数:
            query_text (str): 用户的原始中文/英文问题。
            top_k (int): 最终向 GPT 提供的置信度最高的 Top 页数。
            prefetch_multiplier (int): 第一阶段召回的扩大系数，通常 5 到 10 即可。
            
        返回值里既包含整理后的结果，也包含这一轮检索各阶段的耗时指标。
        """
        resolved_mode = str(retrieval_mode or DEFAULT_RETRIEVAL_MODE).strip().lower()
        if resolved_mode not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

        total_start = time.perf_counter()
        query_colpali_2d: Optional[np.ndarray] = None
        query_muvera_fde: Optional[np.ndarray] = None
        colpali_query_embedding_ms = 0.0
        muvera_query_embedding_ms = 0.0
        colpali_query_token_count = 1

        query_embedding_start = time.perf_counter()
        query_colpali_2d = self._encode_colpali_query(query_text)
        colpali_query_embedding_ms = (time.perf_counter() - query_embedding_start) * 1000
        colpali_query_token_count = max(query_colpali_2d.shape[0], 1)

        if resolved_mode in {"two_stage", "muvera_only"}:
            muvera_query_start = time.perf_counter()
            query_muvera_fde = self._encode_muvera_query(query_colpali_2d)
            muvera_query_embedding_ms = (time.perf_counter() - muvera_query_start) * 1000

        query_embedding_ms = colpali_query_embedding_ms + muvera_query_embedding_ms

        query_filter = self._build_document_filter(document_ids)

        qdrant_query_start = time.perf_counter()
        if resolved_mode == "two_stage":
            if query_muvera_fde is None:
                raise RuntimeError("MUVERA query FDE was not generated for two_stage retrieval.")
            results = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=query_muvera_fde.tolist(),
                        using="muvera_fde",
                        limit=top_k * prefetch_multiplier,
                    )
                ],
                query=query_colpali_2d.tolist(),
                using="original",
                limit=top_k,
                query_filter=query_filter,
            )
            score_normalizer = colpali_query_token_count
            score_space = "colpali_original"
            prefetch_limit = top_k * prefetch_multiplier
        elif resolved_mode == "colpali_only":
            results = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_colpali_2d.tolist(),
                using="original",
                limit=top_k,
                query_filter=query_filter,
            )
            score_normalizer = colpali_query_token_count
            score_space = "colpali_original"
            prefetch_limit = 0
        else:
            if query_muvera_fde is None:
                raise RuntimeError("MUVERA query FDE was not generated for muvera_only retrieval.")
            results = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_muvera_fde.tolist(),
                using="muvera_fde",
                limit=top_k,
                query_filter=query_filter,
            )
            score_normalizer = colpali_query_token_count * MUVERA_SCORE_SCALE
            score_space = "muvera_fde_single"
            prefetch_limit = 0
        qdrant_query_ms = (time.perf_counter() - qdrant_query_start) * 1000

        result_format_start = time.perf_counter()
        unique = self._format_query_results(results.points, top_k=top_k, score_normalizer=score_normalizer)

        result_format_ms = (time.perf_counter() - result_format_start) * 1000
        total_retrieval_ms = (time.perf_counter() - total_start) * 1000
        retrieval_timing = {
            "retrieval_mode": resolved_mode,
            "colpali_query_embedding_ms": round(colpali_query_embedding_ms, 2),
            "muvera_query_embedding_ms": round(muvera_query_embedding_ms, 2),
            "query_embedding_ms": round(query_embedding_ms, 2),
            "qdrant_query_ms": round(qdrant_query_ms, 2),
            "result_format_ms": round(result_format_ms, 2),
            "total_retrieval_ms": round(total_retrieval_ms, 2),
            "prefetch_limit": prefetch_limit,
            "returned_points": len(unique),
            "score_space": score_space,
            "muvera_fde_dimension": self.muvera_fde_dimension,
        }
        logger.info(
            "retrieval_timing mode=%s total_retrieval_ms=%.2f qdrant_query_ms=%.2f query_embedding_ms=%.2f colpali_query_embedding_ms=%.2f muvera_query_embedding_ms=%.2f result_format_ms=%.2f top_k=%s prefetch_limit=%s",
            retrieval_timing["retrieval_mode"],
            retrieval_timing["total_retrieval_ms"],
            retrieval_timing["qdrant_query_ms"],
            retrieval_timing["query_embedding_ms"],
            retrieval_timing["colpali_query_embedding_ms"],
            retrieval_timing["muvera_query_embedding_ms"],
            retrieval_timing["result_format_ms"],
            top_k,
            retrieval_timing["prefetch_limit"],
        )
        print(
            "[Retrieval] "
            f"mode={retrieval_timing['retrieval_mode']} "
            f"total={retrieval_timing['total_retrieval_ms']}ms "
            f"embed={retrieval_timing['query_embedding_ms']}ms "
            f"colpali_embed={retrieval_timing['colpali_query_embedding_ms']}ms "
            f"muvera_embed={retrieval_timing['muvera_query_embedding_ms']}ms "
            f"qdrant={retrieval_timing['qdrant_query_ms']}ms "
            f"format={retrieval_timing['result_format_ms']}ms "
            f"top_k={top_k} prefetch_limit={retrieval_timing['prefetch_limit']}"
        )

        return {"results": unique, "timing": retrieval_timing}
