import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_query_terms(query_text: str) -> List[str]:
    normalized = _normalize_text(query_text)
    terms = set(re.findall(r"[a-z0-9][a-z0-9\-+_]{1,}", normalized))

    # 通用中文短词切片，不绑定任何主题（例如英语）。
    chinese_chars = [ch for ch in query_text if "\u4e00" <= ch <= "\u9fff"]
    for window in (2, 3, 4):
        for idx in range(0, max(0, len(chinese_chars) - window + 1)):
            terms.add("".join(chinese_chars[idx:idx + window]))

    return sorted(term for term in terms if term)


def _score_region(query_terms: List[str], region_text: str) -> float:
    text = _normalize_text(region_text)
    if not text:
        return 0.0

    if not query_terms:
        return 0.0

    score = 0.0
    for term in query_terms:
        if term and term in text:
            score += 1.0 + min(len(term), 8) / 8.0

    return score


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _cache_dir_from_image_path(image_path: str) -> str:
    return os.path.dirname(image_path)


def _metadata_path_from_image_path(image_path: str) -> str:
    return os.path.join(_cache_dir_from_image_path(image_path), "page_regions.json")


def extract_and_cache_page_regions(pdf_path: str, image_paths: List[str], dpi: int = 300) -> Optional[str]:
    """
    从 PDF 原文提取页面文本块和坐标，缓存到每个文档目录下的 page_regions.json。
    失败时返回 None，不影响主链路。
    """
    if not pdf_path or not image_paths:
        return None

    try:
        import fitz  # type: ignore
    except Exception:
        return None

    with fitz.open(pdf_path) as doc:
        page_count = min(len(image_paths), doc.page_count)
        if page_count <= 0:
            return None

        pages_meta: Dict[str, Dict[str, object]] = defaultdict(dict)
        scale = dpi / 72.0

        for page_index in range(page_count):
            page = doc[page_index]
            image_path = image_paths[page_index]
            try:
                with Image.open(image_path) as rendered_image:
                    image_width, image_height = rendered_image.size
            except Exception:
                continue

            page_regions: List[Dict[str, object]] = []
            blocks = page.get_text("blocks") or []
            for block_index, block in enumerate(blocks):
                if len(block) < 7:
                    continue

                x0, y0, x1, y1, text, _, block_type = block[:7]
                if block_type != 0:
                    continue

                region_text = (text or "").strip()
                if not region_text:
                    continue

                left = _clamp(int(round(x0 * scale)) - 12, 0, image_width)
                top = _clamp(int(round(y0 * scale)) - 12, 0, image_height)
                right = _clamp(int(round(x1 * scale)) + 12, 0, image_width)
                bottom = _clamp(int(round(y1 * scale)) + 12, 0, image_height)

                if right - left < 20 or bottom - top < 16:
                    continue

                page_regions.append({
                    "region_id": f"p{page_index + 1}_b{block_index}",
                    "bbox": [left, top, right, bottom],
                    "bbox_norm": [
                        round(left / image_width, 6) if image_width else 0,
                        round(top / image_height, 6) if image_height else 0,
                        round(right / image_width, 6) if image_width else 0,
                        round(bottom / image_height, 6) if image_height else 0,
                    ],
                    "text": region_text,
                })

            pages_meta[str(page_index + 1)] = {
                "image_size": [image_width, image_height],
                "regions": page_regions,
            }

    metadata = {"dpi": dpi, "pages": pages_meta}
    metadata_path = _metadata_path_from_image_path(image_paths[0])
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata_path


def load_page_regions_for_image(image_path: str) -> Tuple[List[Dict[str, object]], List[int]]:
    metadata_path = _metadata_path_from_image_path(image_path)
    if not os.path.exists(metadata_path):
        return [], []

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return [], []

    page_number = os.path.splitext(os.path.basename(image_path))[0].split("_")[-1]
    pages = metadata.get("pages", {}) if isinstance(metadata, dict) else {}
    page_meta = pages.get(str(page_number), {}) if isinstance(pages, dict) else {}

    # 兼容旧格式: page_meta 可能直接是 regions 列表。
    if isinstance(page_meta, list):
        regions = [region for region in page_meta if isinstance(region, dict)]
        return regions, []

    if not isinstance(page_meta, dict):
        return [], []

    regions_raw = page_meta.get("regions", [])
    image_size_raw = page_meta.get("image_size", [])

    regions = [region for region in regions_raw if isinstance(region, dict)] if isinstance(regions_raw, list) else []
    image_size = image_size_raw if isinstance(image_size_raw, list) and len(image_size_raw) == 2 else []

    return regions, image_size


def _page_level_evidence(result: Dict[str, object], image_size: Optional[List[int]] = None) -> Dict[str, object]:
    return {
        "document_name": result.get("document_name", "Unknown File"),
        "page_number": result.get("page_number", 0),
        "score": float(result.get("score", 0.0)),
        "image_path": result.get("image_path", ""),
        "image_kind": "page",
        "image_size": image_size or [],
        "regions": [],
    }


def build_localized_evidences(
    query_text: str,
    results: List[Dict[str, object]],
    max_total: int = 6,
    max_regions_per_page: int = 4,
    min_region_score: float = 0.5,
    extra_terms: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """
    基于页级召回结果，返回"原页图 + 高亮框列表"的证据结构。

    extra_terms: 来自 LLM 答案的附加词项，与 query_terms 合并后一起用于区域打分。
    通过"答案驱动高亮"机制弥补 query 措辞与 PDF 原文之间的语义鸿沟。
    """
    if not results:
        return []

    query_terms = list(set(_extract_query_terms(query_text) + (extra_terms or [])))
    evidences: List[Dict[str, object]] = []
    seen_pages = set()

    for result in results:
        image_path = str(result.get("image_path", ""))
        page_number = int(result.get("page_number", 0) or 0)
        page_key = (result.get("document_id", ""), page_number, image_path)
        if page_key in seen_pages:
            continue

        seen_pages.add(page_key)
        page_regions, image_size = load_page_regions_for_image(image_path)

        if not page_regions:
            evidences.append(_page_level_evidence(result, image_size))
            if len(evidences) >= max_total:
                break
            continue

        ranked_regions: List[Tuple[float, Dict[str, object]]] = []
        for region in page_regions:
            region_text = str(region.get("text", ""))
            region_score = _score_region(query_terms, region_text)
            if region_score >= min_region_score:
                ranked_regions.append((region_score, region))

        ranked_regions.sort(key=lambda item: item[0], reverse=True)

        selected_regions: List[Dict[str, object]] = []
        for region_score, region in ranked_regions[:max_regions_per_page]:
            bbox_norm = region.get("bbox_norm")
            if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
                continue
            selected_regions.append({
                "region_id": region.get("region_id"),
                "bbox_norm": [float(v) for v in bbox_norm],
                "text": str(region.get("text", "")),
                "region_score": float(region_score),
            })

        # 标题块提升：对已匹配的描述块，将其前驱块（论文/项目标题行）也纳入高亮
        # 规则：前驱块文本长度在 10-300 之间（标题行特征），且尚未被选中
        selected_ids = {r["region_id"] for r in selected_regions}
        promoted: List[Dict[str, object]] = []
        for sel_region in list(selected_regions):
            rid = str(sel_region.get("region_id") or "")
            m = re.match(r"p(\d+)_b(\d+)", rid)
            if not m:
                continue
            page_num_str = m.group(1)
            block_idx = int(m.group(2))
            if block_idx == 0:
                continue
            pred_rid = f"p{page_num_str}_b{block_idx - 1}"
            if pred_rid in selected_ids:
                continue
            pred_region = next((pr for pr in page_regions if pr.get("region_id") == pred_rid), None)
            if pred_region is None:
                continue
            pred_text = str(pred_region.get("text", "")).strip()
            if 10 <= len(pred_text) <= 300:
                bbox_norm = pred_region.get("bbox_norm")
                if isinstance(bbox_norm, list) and len(bbox_norm) == 4:
                    promoted.append({
                        "region_id": pred_rid,
                        "bbox_norm": [float(v) for v in bbox_norm],
                        "text": pred_text,
                        "region_score": sel_region.get("region_score", 1.0) * 0.9,
                    })
                    selected_ids.add(pred_rid)

        if promoted:
            # 标题提升结果独立追加，不与 selected_regions 竞争配额
            # 总上限为 max_regions_per_page + 2，确保标题行不被 bullet 挤掉
            combined = selected_regions + promoted
            combined.sort(key=lambda r: r.get("region_score", 0), reverse=True)
            selected_regions = combined[:max_regions_per_page + 2]

        evidence = _page_level_evidence(result, image_size)
        evidence["regions"] = selected_regions
        evidences.append(evidence)

        if len(evidences) >= max_total:
            break

    return evidences[:max_total]
