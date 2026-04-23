import hashlib
from typing import Any, Dict, List, Optional
import warnings

# 过滤并忽略 PyTorch 在部分系统中无伤大雅的类发现警告
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

from qdrant_client import QdrantClient, models
from fastembed.postprocess.muvera import Muvera
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image

from src.config import QDRANT_URL, COLLECTION_NAME, COLPALI_MODEL_NAME

class VisionVectorStore:
    """
    视觉向量数据库服务类，封装了 ColPali 多模特征模型、MUVERA 聚类压缩及 Qdrant 本地数据库操作。
    主要职责：
    1. 使用预训练视觉-语言大模型（ColPali）将包含文本图片的文档页转化为稠密多维度特征（Multi-Vector Embedding）。
    2. 计算 MUVERA（聚类降维表示）加速在大规模文档下的第一阶段海选检索（Prefetch）。
    3. 连接并操作 Qdrant，保障两阶段检索流程。
    """
    def __init__(self):
        # 1. 链接至本地 Docker 容器启动的 Qdrant 数据库
        # check_compatibility=False 屏蔽客户端与服务端的版本次要差异警告
        self.qdrant = QdrantClient(url=QDRANT_URL, check_compatibility=False)
        
        # 2. 定义 MUVERA 实例进行维度压缩与计算加速
        # 这是一种针对 Late-Interaction 模型的多维聚类方式，用于缓解在海量文档下搜索时的性能瓶颈
        self.muvera = Muvera(
            dim=128,          # ColPali 原始维度 128
            k_sim=6,          # 聚类数目 (2^6 = 64 clusters)
            dim_proj=16,      # 每一簇压缩至 16 维
            r_reps=30,        # 重复 30 次随机投影（提升 Stage 1 近似精度，CPU 侧额外开销 <3ms）
            random_seed=42,   # 固定随机种子确保线上复现性
        )
        
        # 3. 初始化 ColPali 视觉语言大模型
        # 判断当前硬件设备，具备 GPU 则用 GPU 加速推理
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_NAME)
        # 为节约本地显存，强烈建议在 Mac mps 或者普通 CPU 下使用 bfloat16 以避免 OOM：
        dtype = torch.bfloat16
        
        # 加上 low_cpu_mem_usage=True 是为防止一次性预热分配过大连续显存导致崩溃
        self.model = ColPali.from_pretrained(
            COLPALI_MODEL_NAME,
            torch_dtype=dtype,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        
        # 确保名为 colpali-rag-collection 的业务集合已建立，供入库和查询使用
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        初始化核心的 Collection 结构。
        在使用多向量（ColPali机制）时，为了实现前文提及的“两阶段检索”，这里必须定义两种不同结构的向量池：
        1) "original": 完整的、未压缩的 ColPali 视觉多维表示（高精度，但扫描慢）。
        2) "muvera_fde": 经过 MUVERA 降维策略的压缩特征（精度略损耗，但计算快）。
        """
        if not self.qdrant.collection_exists(COLLECTION_NAME):
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    # 原生精确匹配向量 (ColPali 输出的尺寸，使用多向量模式 MULTIVECTOR)
                    "original": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                    # MUVERA 第一阶段快速召回（Prefetch）紧凑向量 
                    "muvera_fde": models.VectorParams(
                        size=16,
                        distance=models.Distance.DOT,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                }
            )

    def reset_collection(self):
        """
        完全清空并重建当前的 Qdrant 集合。
        主要用于每次项目重启时清除全部数据，避免状态膨胀。
        """
        if self.qdrant.collection_exists(COLLECTION_NAME):
            self.qdrant.delete_collection(COLLECTION_NAME)
        self._ensure_collection_exists()

    def _make_point_id(self, document_id: str, page_number: int) -> int:
        """
        通过 document_id + page_number 生成稳定且可复现的 63-bit 整数 ID，避免哈希碰撞覆盖。
        """
        digest = hashlib.sha1(f"{document_id}:{page_number}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16) & ((1 << 63) - 1)

    def _build_document_filter(self, document_ids: Optional[List[str]]) -> Optional[models.Filter]:
        """
        根据文档 ID 列表构建 Qdrant 过滤器；为空时表示全库检索。
        """
        if not document_ids:
            return None

        clean_ids = sorted({doc_id for doc_id in document_ids if doc_id})
        if not clean_ids:
            return None

        if len(clean_ids) == 1:
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=clean_ids[0]),
                    )
                ]
            )

        return models.Filter(
            should=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=doc_id),
                )
                for doc_id in clean_ids
            ]
        )

    def delete_document(self, document_id: str):
        """
        按文档 ID 删除历史向量，确保同一文档重复上传时不会累计脏数据。
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
        **_ignored,
    ) -> bool:
        """
        将转换好的 PDF 每页图片利用 ColPali 模型送入并转为稠密向量，通过 MUVERA 进行压缩扩展，然后写到 Qdrant 中。
        采用分批处理（Batched Processing），防止由于页数太多导致本地显存或系统内存爆炸。
        
        参数:
            image_paths (List[str]): 用于处理和存储的多页图片的物理路径。
            batch_size (int): 每次送入模型的并行图片数，默认为 4。
        """
        document_id = document_id or hashlib.sha1("\n".join(image_paths).encode("utf-8")).hexdigest()
        document_name = document_name or document_id

        if replace_document:
            self.delete_document(document_id)

        total_images = len(image_paths)
        points = []
        
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            try:
                # 将图片信息预处理为 ColPali 模型可接收的格式 (类似 Vit 结合 LLaMA 预处理)
                batch_inputs = self.processor.process_images(images).to(self.device).to(self.model.dtype)

                with torch.no_grad():
                    # 获取高质量多向量输出
                    image_embeddings = self.model(**batch_inputs)
            finally:
                for img in images:
                    img.close()
                
            # 转换为 Numpy，接着使用 MUVERA 根据原本的多模态特征派生压缩版的加速特征
            cpu_embeddings = [emb.float().cpu().numpy() for emb in image_embeddings]
            
            for j, (path, original_emb) in enumerate(zip(batch_paths, cpu_embeddings)):
                # MUVERA 生成的结构可能受输入分辨率或页数影响，变为铺平的一维巨型数组
                # 所以无论如何都在入库前强制经过防御性重塑 (Defensive Reshape) 到目标列维
                original_emb_2d = original_emb.reshape(-1, 128)
                muvera_embRaw = self.muvera.process_document(original_emb)
                muvera_emb_2d = muvera_embRaw.reshape(-1, 16)
                
                # 使用页码作为稳定索引，避免多文档混写时覆盖。
                page_number = i + j + 1
                point = models.PointStruct(
                    id=self._make_point_id(document_id=document_id, page_number=page_number),
                    vector={
                        "original": original_emb_2d.tolist(),
                        "muvera_fde": muvera_emb_2d.tolist(),
                    },
                    payload={
                        "image_path": path,
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_number,
                    }
                )
                points.append(point)
                
        if not points:
            return False

        # 分批 upsert，每批 8 个点，避免单次请求体超过 Qdrant 默认 32MB 上限
        # 每页向量 JSON 约 1.94 MB（original 1.57 MB + muvera 0.37 MB），8 页 ≈ 15.5 MB，留足余量
        upsert_batch_size = 8
        for i in range(0, len(points), upsert_batch_size):
            self.qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i + upsert_batch_size]
            )
        return True

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        获取当前向量库中所有已装载的独立文件列表。
        通过聚合 payload 中的 document_id 和 document_name 实现。
        """
        try:
            records, _ = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
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

    def retrieve_with_two_stage(
        self,
        query_text: str,
        top_k: int = 3,
        prefetch_multiplier: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        用户输入问题时，通过 ColPali 获取 Text Embedding，
        利用 Qdrant 第一级 "muvera_fde" 扩大召回返回大基数侯选人（Prefetch），
        随后第二级即利用 "original" 多向量通过 Max-Sim 的运算执行精准重排（Rerank）。
        这就是多模态检索提速 10x 以上的 RAG 核心技巧。
        
        参数:
            query_text (str): 用户的原始中文/英文问题。
            top_k (int): 最终向 GPT 提供的置信度最高的 Top 页数。
            prefetch_multiplier (int): 第一阶段召回的扩大系数，通常 5 到 10 即可。
            
        返回:
            List[Tuple[image_path, score]]: 匹配度排名前 k 的包含路径列表与权重的二元组。
        """
        # 第一步：计算用户查询的问题向量
        batch_queries = self.processor.process_queries([query_text]).to(self.device)
        with torch.no_grad():
            # 同样输出高维度的 128D Multi-vector
            query_embedding_tensor = self.model(**batch_queries)[0]
        
        query_colpali = query_embedding_tensor.float().cpu().numpy()
        # 同样生成问题的 MUVERA 压缩版以便和图片 MUVERA 比对
        query_muvera = self.muvera.process_query(query_colpali)

        # 同样执行防御性维度重塑 (Defensive Reshape)，防止一维铺平错误
        query_colpali_2d = query_colpali.reshape(-1, 128)
        query_muvera_2d = query_muvera.reshape(-1, 16)
        # 归一化因子：查询 token 数量，使分数落入 [0, 1] 区间，消除查询长度对阈值的影响
        n_query_tokens = max(query_colpali_2d.shape[0], 1)

        query_filter = self._build_document_filter(document_ids)

        # 第二步：单次 API 请求完成"海选"加"优选"
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_muvera_2d.tolist(),
                    using="muvera_fde",      # 用压缩版做海量打分过滤
                    limit=top_k * prefetch_multiplier,
                )
            ],
            query=query_colpali_2d.tolist(),
            using="original",                # 从上面过滤出的一小部分数据里做精确计算 (MaxSim 距离)
            limit=top_k,
            query_filter=query_filter,
        )

        # 对结果进行去重，避免 1 页文档重复返回多次相同证据页。
        unique = []
        seen = set()
        for point in results.points:
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
                    "score": point.score / n_query_tokens,
                    "document_id": payload.get("document_id", ""),
                    "document_name": payload.get("document_name", ""),
                    "page_number": payload.get("page_number", 0),
                }
            )
            if len(unique) >= top_k:
                break

        return unique
