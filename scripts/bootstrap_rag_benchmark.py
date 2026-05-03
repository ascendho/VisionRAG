#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from xml.etree import ElementTree as ET

import fitz
import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARK_API_KEY, DOUBAO_MODEL_NAME
from src.doc_processor import get_file_hash


SLIDE_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
SUPPORTED_SUFFIXES = {".pdf", ".pptx"}
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when", "where", "which",
    "一个", "一种", "这个", "这页", "页面", "内容", "根据", "说明", "主要", "关于", "以及", "进行",
}


@dataclass
class PageUnit:
    document_path: Path
    document_name: str
    document_id: str
    source_type: str
    page_number: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a small RAG benchmark draft from local PDF/PPTX files.",
    )
    parser.add_argument("--source-dir", default="benchmarks/sample-documents", help="Directory containing source PDF/PPTX files.")
    parser.add_argument("--documents", nargs="*", help="Specific file names to use from the source directory.")
    parser.add_argument("--max-documents", type=int, default=2, help="How many documents to use when --documents is omitted.")
    parser.add_argument("--target-question-count", type=int, default=10, help="Target number of benchmark questions to generate.")
    parser.add_argument("--questions-per-page", type=int, default=1, help="Maximum questions to generate for each selected page.")
    parser.add_argument("--max-pages-per-document", type=int, default=8, help="Maximum candidate pages to inspect per document.")
    parser.add_argument("--min-page-chars", type=int, default=120, help="Minimum extracted characters required for a page to be used.")
    parser.add_argument(
        "--generator-mode",
        choices=["auto", "llm", "heuristic"],
        default="auto",
        help="Question generation mode. auto prefers LLM and falls back to heuristics.",
    )
    parser.add_argument("--llm-model", default=DOUBAO_MODEL_NAME, help="Model used when generator mode includes LLM.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000", help="Optional backend URL used only to warn about missing indexed docs.")
    parser.add_argument("--no-index-check", action="store_true", help="Skip checking whether selected docs are already indexed.")
    parser.add_argument("--output-file", default="benchmarks/rag_eval_small_draft.json", help="Path to the generated benchmark JSON.")
    parser.add_argument("--review-csv", help="Optional review CSV path. Defaults next to the output JSON.")
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
    filtered = [line for line in lines if line]
    return "\n".join(filtered)


def normalize_key(text: str) -> str:
    return re.sub(r"\W+", "", text.lower())


def normalize_match_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", "", lowered)
    lowered = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", lowered)
    return lowered


def answer_is_grounded(page_text: str, gold_answer: str, keywords: Sequence[str]) -> bool:
    normalized_page = normalize_match_text(page_text)
    normalized_answer = normalize_match_text(gold_answer)
    if 0 < len(normalized_answer) < 2:
        return False
    if normalized_answer and normalized_answer in normalized_page:
        return True

    for keyword in keywords:
        normalized_keyword = normalize_match_text(str(keyword))
        if len(normalized_keyword) >= 2 and normalized_keyword in normalized_page:
            return True
    return False


def build_relevant_excerpt(page_text: str, gold_answer: str, keywords: Sequence[str], window: int = 260) -> str:
    compact_text = re.sub(r"\s+", " ", page_text).strip()
    if not compact_text:
        return ""

    search_terms = sorted(
        [gold_answer, *[str(keyword) for keyword in keywords]],
        key=lambda term: len(str(term).strip()),
        reverse=True,
    )
    lowered_text = compact_text.lower()
    for term in search_terms:
        normalized_term = str(term).strip()
        if not normalized_term:
            continue
        position = lowered_text.find(normalized_term.lower())
        if position >= 0:
            start = max(0, position - window // 3)
            end = min(len(compact_text), position + len(normalized_term) + (window * 2) // 3)
            return compact_text[start:end]

    return compact_text[:window]


def read_source_documents(source_dir: Path, requested_docs: Optional[Sequence[str]], max_documents: int) -> List[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if requested_docs:
        resolved: List[Path] = []
        for name in requested_docs:
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = source_dir / name
            if not candidate.exists():
                raise FileNotFoundError(f"Requested document not found: {candidate}")
            if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported document type: {candidate.name}")
            resolved.append(candidate)
        return resolved

    files = [
        path for path in sorted(source_dir.iterdir(), key=lambda item: (item.suffix.lower() != ".pdf", item.name.lower()))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files[:max_documents]


def extract_pdf_pages(document_path: Path) -> List[PageUnit]:
    document_id = get_file_hash(str(document_path))
    pages: List[PageUnit] = []
    with fitz.open(document_path) as doc:
        for index, page in enumerate(doc, start=1):
            text = normalize_whitespace(page.get_text("text"))
            pages.append(
                PageUnit(
                    document_path=document_path,
                    document_name=document_path.name,
                    document_id=document_id,
                    source_type="pdf",
                    page_number=index,
                    text=text,
                )
            )
    return pages


def slide_number(slide_path: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", slide_path)
    return int(match.group(1)) if match else 0


def extract_pptx_pages(document_path: Path) -> List[PageUnit]:
    document_id = get_file_hash(str(document_path))
    pages: List[PageUnit] = []
    with zipfile.ZipFile(document_path) as archive:
        slide_files = sorted(
            [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")],
            key=slide_number,
        )
        for slide_path in slide_files:
            root = ET.fromstring(archive.read(slide_path))
            parts = [node.text.strip() for node in root.findall(".//a:t", SLIDE_NS) if node.text and node.text.strip()]
            text = normalize_whitespace("\n".join(parts))
            pages.append(
                PageUnit(
                    document_path=document_path,
                    document_name=document_path.name,
                    document_id=document_id,
                    source_type="pptx",
                    page_number=slide_number(slide_path),
                    text=text,
                )
            )
    return pages


def extract_document_pages(document_path: Path) -> List[PageUnit]:
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(document_path)
    if suffix == ".pptx":
        return extract_pptx_pages(document_path)
    raise ValueError(f"Unsupported document type: {document_path}")


def split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[。！？!?\.])\s+", normalized) if part.strip()]


def detect_heading(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip(" -•\t")
        if 4 <= len(candidate) <= 120:
            return candidate
    sentences = split_sentences(text)
    return sentences[0][:120] if sentences else ""


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    tokens = re.findall(r"\d+(?:\.\d+)?|[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", text)
    keywords: List[str] = []
    seen = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def choose_fact_sentence(text: str, heading: str) -> str:
    sentences = split_sentences(text)
    heading_key = normalize_key(heading)
    numeric_candidates = [sentence for sentence in sentences if re.search(r"\d", sentence)]
    for sentence in numeric_candidates + sentences:
        if len(sentence) < 18:
            continue
        if heading_key and normalize_key(sentence) == heading_key:
            continue
        return sentence[:180]
    return ""


def score_page(page: PageUnit) -> int:
    heading = detect_heading(page.text)
    number_count = len(re.findall(r"\d", page.text))
    return len(page.text) + len(heading) * 2 + number_count * 20


def select_focus_pages(pages: Sequence[PageUnit], min_page_chars: int, max_pages_per_document: int) -> List[PageUnit]:
    usable = [page for page in pages if len(re.sub(r"\s+", "", page.text)) >= min_page_chars]
    if len(usable) <= max_pages_per_document:
        return list(usable)

    ranked = sorted(usable, key=lambda page: (-score_page(page), page.page_number))
    selected: List[PageUnit] = []
    for page in ranked:
        if len(selected) >= max_pages_per_document:
            break
        if all(abs(page.page_number - existing.page_number) >= 2 for existing in selected):
            selected.append(page)
    if len(selected) < max_pages_per_document:
        for page in ranked:
            if page in selected:
                continue
            selected.append(page)
            if len(selected) >= max_pages_per_document:
                break
    return sorted(selected, key=lambda page: page.page_number)


def interleave_pages(document_pages: Sequence[Sequence[PageUnit]]) -> Iterable[PageUnit]:
    max_len = max((len(pages) for pages in document_pages), default=0)
    for index in range(max_len):
        for pages in document_pages:
            if index < len(pages):
                yield pages[index]


def build_llm_client(api_key: str) -> OpenAI:
    return OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=api_key)


def parse_llm_json_array(raw_text: str) -> List[Dict[str, Any]]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned).strip()
    for candidate in (cleaned, cleaned[cleaned.find("["): cleaned.rfind("]") + 1] if "[" in cleaned and "]" in cleaned else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            return [item for item in parsed["items"] if isinstance(item, dict)]
    return []


def llm_generate_candidates(page: PageUnit, max_questions: int, llm_model: str) -> List[Dict[str, Any]]:
    if not ARK_API_KEY:
        return []

    prompt_text = page.text[:2800]
    client = build_llm_client(ARK_API_KEY)
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "你在帮助构造视觉 RAG benchmark。"
                    "你只能根据当前单页文本生成问题和标准答案，不能使用页外知识。"
                    "问题必须满足：单页可答、容易判分、答案尽量短、不要依赖上下文代词、不要生成是非题。"
                    "优先生成数值查找、标题主题、定义总结、列表要点、限制条件这类题。"
                    f"请输出不超过 {max_questions} 个对象组成的 JSON 数组，每个对象只包含 question, question_type, gold_answer, gold_answer_keywords。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"文档名：{page.document_name}\n"
                    f"页码：{page.page_number}\n"
                    "请只基于下面这段单页文本生成可判分的问答对。\n\n"
                    f"{prompt_text}"
                ),
            },
        ],
    )
    raw_text = response.choices[0].message.content or ""
    parsed = parse_llm_json_array(raw_text)
    candidates: List[Dict[str, Any]] = []
    for item in parsed:
        question = str(item.get("question") or "").strip()
        gold_answer = str(item.get("gold_answer") or "").strip()
        if not question or not gold_answer:
            continue
        if 0 < len(normalize_match_text(gold_answer)) < 2:
            continue
        raw_keywords = item.get("gold_answer_keywords") or []
        if isinstance(raw_keywords, str):
            raw_keywords = re.split(r"[，,、/;；]\s*", raw_keywords)
        keywords = [str(keyword).strip() for keyword in raw_keywords if str(keyword).strip()]
        if not keywords:
            keywords = extract_keywords(gold_answer)
        if not answer_is_grounded(page.text, gold_answer, keywords):
            continue
        candidates.append(
            {
                "question": question,
                "question_type": str(item.get("question_type") or "single_page_fact").strip() or "single_page_fact",
                "gold_answer": gold_answer,
                "gold_answer_keywords": keywords[:5],
                "generation_mode": "llm",
            }
        )
    return candidates[:max_questions]


def heuristic_generate_candidates(page: PageUnit, max_questions: int) -> List[Dict[str, Any]]:
    heading = detect_heading(page.text)
    fact_sentence = choose_fact_sentence(page.text, heading)
    candidates: List[Dict[str, Any]] = []

    if heading:
        candidates.append(
            {
                "question": f"{page.document_name} 第 {page.page_number} 页的标题或主题是什么？",
                "question_type": "title_lookup",
                "gold_answer": heading,
                "gold_answer_keywords": extract_keywords(heading),
                "generation_mode": "heuristic",
            }
        )

    if fact_sentence and len(candidates) < max_questions:
        candidates.append(
            {
                "question": f"根据 {page.document_name} 第 {page.page_number} 页，这一页明确给出的一个关键信息是什么？",
                "question_type": "single_page_fact",
                "gold_answer": fact_sentence,
                "gold_answer_keywords": extract_keywords(fact_sentence),
                "generation_mode": "heuristic",
            }
        )

    return candidates[:max_questions]


def resolve_generation_mode(requested_mode: str) -> str:
    if requested_mode == "llm":
        return "llm"
    if requested_mode == "heuristic":
        return "heuristic"
    return "llm" if ARK_API_KEY else "heuristic"


def generate_candidates(page: PageUnit, requested_mode: str, max_questions: int, llm_model: str) -> List[Dict[str, Any]]:
    mode = resolve_generation_mode(requested_mode)
    if mode == "llm":
        try:
            candidates = llm_generate_candidates(page, max_questions, llm_model)
        except Exception:
            candidates = []
        if candidates:
            return candidates
    return heuristic_generate_candidates(page, max_questions)


def build_item(index: int, page: PageUnit, candidate: Dict[str, Any]) -> Dict[str, Any]:
    excerpt = build_relevant_excerpt(
        page.text,
        str(candidate.get("gold_answer") or ""),
        list(candidate.get("gold_answer_keywords") or []),
    )
    return {
        "id": f"auto-{index:03d}",
        "question": str(candidate.get("question") or "").strip(),
        "question_type": str(candidate.get("question_type") or "single_page_fact").strip() or "single_page_fact",
        "document_ids": [page.document_id],
        "gold_evidence": [
            {
                "document_id": page.document_id,
                "document_name": page.document_name,
                "page_number": page.page_number,
            }
        ],
        "gold_answer": str(candidate.get("gold_answer") or "").strip(),
        "gold_answer_keywords": list(candidate.get("gold_answer_keywords") or [])[:5],
        "notes": f"自动起草草稿，请复核答案是否可由单页直接得出。来源模式：{candidate.get('generation_mode', 'heuristic')}。",
        "source_excerpt": excerpt,
        "generation_mode": candidate.get("generation_mode", "heuristic"),
    }


def validate_generated_payload(payload: Dict[str, Any]) -> None:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Generated payload must contain a non-empty items list.")
    for index, item in enumerate(items, start=1):
        if not str(item.get("id") or "").strip():
            raise ValueError(f"Generated item #{index} is missing id.")
        if not str(item.get("question") or "").strip():
            raise ValueError(f"Generated item #{index} is missing question.")
        if not str(item.get("gold_answer") or "").strip():
            raise ValueError(f"Generated item #{index} is missing gold_answer.")
        gold_evidence = item.get("gold_evidence")
        if not isinstance(gold_evidence, list) or not gold_evidence:
            raise ValueError(f"Generated item #{index} is missing gold_evidence.")
        gold = gold_evidence[0]
        if int(gold.get("page_number") or 0) <= 0:
            raise ValueError(f"Generated item #{index} has invalid page_number.")
        if not str(gold.get("document_id") or gold.get("document_name") or "").strip():
            raise ValueError(f"Generated item #{index} is missing document identity.")


def derive_review_csv_path(output_file: Path, explicit_review_csv: Optional[str]) -> Path:
    if explicit_review_csv:
        return Path(explicit_review_csv)
    return output_file.with_name(f"{output_file.stem}_review.csv")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_review_csv(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "document_name",
        "document_id",
        "page_number",
        "question_type",
        "question",
        "gold_answer",
        "gold_answer_keywords",
        "generation_mode",
        "source_excerpt",
        "notes",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            gold = (item.get("gold_evidence") or [{}])[0]
            writer.writerow(
                {
                    "id": item.get("id", ""),
                    "document_name": gold.get("document_name", ""),
                    "document_id": gold.get("document_id", ""),
                    "page_number": gold.get("page_number", ""),
                    "question_type": item.get("question_type", ""),
                    "question": item.get("question", ""),
                    "gold_answer": item.get("gold_answer", ""),
                    "gold_answer_keywords": json.dumps(item.get("gold_answer_keywords") or [], ensure_ascii=False),
                    "generation_mode": item.get("generation_mode", ""),
                    "source_excerpt": item.get("source_excerpt", ""),
                    "notes": item.get("notes", ""),
                    "reviewer_notes": "",
                }
            )


def fetch_indexed_document_ids(api_base_url: str) -> Optional[set[str]]:
    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/api/rag/files", timeout=5)
        response.raise_for_status()
    except Exception:
        return None
    payload = response.json()
    files = payload.get("files") or []
    return {str(item.get("document_id") or "").strip() for item in files if str(item.get("document_id") or "").strip()}


def build_payload(items: Sequence[Dict[str, Any]], documents: Sequence[Path], generation_mode: str, indexed_document_ids: Optional[set[str]]) -> Dict[str, Any]:
    metadata_docs = []
    for document in documents:
        document_id = get_file_hash(str(document))
        metadata_docs.append(
            {
                "document_name": document.name,
                "document_id": document_id,
                "indexed_in_backend": None if indexed_document_ids is None else document_id in indexed_document_ids,
            }
        )
    return {
        "version": "1.0",
        "description": "自动起草的小规模视觉 RAG benchmark 草稿，请在跑分前复核问题、答案与页码。",
        "defaults": {
            "top_k": 5,
            "min_score": 0.6,
            "chat_history": [],
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "requested_generation_mode": generation_mode,
            "documents": metadata_docs,
        },
        "items": list(items),
    }


def main() -> int:
    args = parse_args()
    source_dir = (PROJECT_ROOT / args.source_dir).resolve()
    documents = read_source_documents(source_dir, args.documents, args.max_documents)
    if not documents:
        raise SystemExit("No supported PDF/PPTX files were found.")

    indexed_document_ids = None if args.no_index_check else fetch_indexed_document_ids(args.api_base_url)

    focus_page_groups: List[List[PageUnit]] = []
    for document in documents:
        pages = extract_document_pages(document)
        focus_pages = select_focus_pages(pages, args.min_page_chars, args.max_pages_per_document)
        if focus_pages:
            focus_page_groups.append(focus_pages)

    if not focus_page_groups:
        raise SystemExit("No usable pages were extracted. Try lowering --min-page-chars or choosing different files.")

    target_count = max(1, args.target_question_count)
    questions_per_page = max(1, args.questions_per_page)
    generated_items: List[Dict[str, Any]] = []
    seen_questions = set()

    for page in interleave_pages(focus_page_groups):
        candidates = generate_candidates(page, args.generator_mode, questions_per_page, args.llm_model)
        for candidate in candidates:
            question_key = (page.document_id, page.page_number, normalize_key(str(candidate.get("question") or "")))
            if question_key in seen_questions:
                continue
            item = build_item(len(generated_items) + 1, page, candidate)
            generated_items.append(item)
            seen_questions.add(question_key)
            if len(generated_items) >= target_count:
                break
        if len(generated_items) >= target_count:
            break

    if not generated_items:
        raise SystemExit("Failed to generate any benchmark items.")

    if len(generated_items) < target_count:
        print(
            f"Warning: only generated {len(generated_items)} item(s) out of requested {target_count}. "
            "Try adding more documents or increasing --max-pages-per-document.",
            file=sys.stderr,
        )

    payload = build_payload(
        generated_items,
        documents,
        resolve_generation_mode(args.generator_mode),
        indexed_document_ids,
    )
    validate_generated_payload(payload)

    output_file = Path(args.output_file)
    review_csv = derive_review_csv_path(output_file, args.review_csv)
    write_json(output_file, payload)
    write_review_csv(review_csv, generated_items)

    print(f"Generated {len(generated_items)} benchmark item(s).")
    print(f"Benchmark JSON: {output_file}")
    print(f"Review CSV: {review_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())