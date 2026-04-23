import os
import shutil
import hashlib
import textwrap
from typing import List
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from src.config import IMAGE_CACHE_DIR

# ── 文本渲染配置 ──────────────────────────────────────────────
_TEXT_PAGE_W = 850
_TEXT_PAGE_H = 1200
_TEXT_MARGIN = 60       # px，四周留白
_TEXT_FONT_SIZE = 18
_TEXT_LINE_HEIGHT = 30  # px

def _get_font(size: int = _TEXT_FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """尝试加载支持 CJK 的系统字体，失败则回退到 Pillow 内置字体。"""
    candidates = [
        # macOS — CJK 优先
        "/System/Library/Fonts/PingFang.ttc",               # 苹方（macOS 10.11+，完整 CJK）
        "/System/Library/Fonts/STHeiti Medium.ttc",          # 华文黑体中等
        "/System/Library/Fonts/STHeiti Light.ttc",           # 华文黑体细体
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux — CJK
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # 最后兜底（ASCII 正常，无 CJK）
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Pillow 10+ 的 load_default 支持 size 参数
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def clear_all_caches() -> None:
    """
    清理所有文档的图片缓存目录。
    """
    if os.path.exists(IMAGE_CACHE_DIR):
        try:
            shutil.rmtree(IMAGE_CACHE_DIR)
        except Exception as e:
            print(f"清理图片缓存失败: {e}")
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """
    计算文件内容的 MD5 哈希值。
    主要用于判断用户是否上传过同一份 PDF 文件，避免重复解析。
    
    参数:
        file_path (str): 文件路径
        
    返回:
        str: 32位 MD5 字符串
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取文件，支持读取大文件
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[str]:
    """
    将 PDF 文件的每一页分别转为高质量 PNG 截图并保存到缓存目录。
    这个步骤是整个多模态 RAG 流程的基石：
    传统的文本 RAG 将 PDF 提取为纯文本，会丢失图表、排版等视觉信息。
    而 ColPali 视觉语言模型天然支持图文理解，我们将其转换为整页图片可以极大保留原貌。
    
    参数:
        pdf_path (str): PDF 文件的绝对/相对路径。
        dpi (int): 分辨率（默认为300），越高图像越清晰，但处理时间、文件大小会明显增大。
        
    返回:
        List[str]: 所有页面对应的图片保存路径列表。
    """
    file_hash = get_file_hash(pdf_path)
    # 为当前文件创建独立的缓存目录，防止重名污染
    doc_cache_dir = os.path.join(IMAGE_CACHE_DIR, file_hash)
    os.makedirs(doc_cache_dir, exist_ok=True)
    
    try:
        # 解析 PDF 得到 PIL 图像对象列表
        # (需要提前在系统安装 poppler，例如 Mac 上运行: brew install poppler)
        pages = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    except Exception as e:
        raise RuntimeError(f"PDF 文档 {pdf_path} 读取失败，请检查是否为受损文件或系统是否安装了 poppler. 错误信息: {str(e)}")

    image_paths = []
    # 遍历页数，分别存盘。如果检测到本地已有图片缓存则略过，起到提速效果
    for i, page in enumerate(pages):
        image_name = f"page_{i + 1}.png"
        image_path = os.path.join(doc_cache_dir, image_name)
        
        if not os.path.exists(image_path):
            page.save(image_path, "PNG")
            
        # 记录图像路径供后续特征提取以及前端查看时使用
        image_paths.append(image_path)
        
    return image_paths


def process_image_to_images(image_path: str) -> List[str]:
    """
    处理单张图片文件（PNG/JPG/JPEG/WEBP），将其作为"单页文档"纳入索引。
    直接跳过 PDF 转图步骤，将图片缓存到统一目录后返回路径列表。
    
    参数:
        image_path (str): 图片文件路径
        
    返回:
        List[str]: 包含缓存后图片路径的单元素列表
    """
    file_hash = get_file_hash(image_path)
    doc_cache_dir = os.path.join(IMAGE_CACHE_DIR, file_hash)
    os.makedirs(doc_cache_dir, exist_ok=True)

    # 统一缓存为 PNG，避免 JPG/WEBP 等格式在 ColPali 推理时出现 RGBA/P 模式问题
    cached_path = os.path.join(doc_cache_dir, "page_1.png")

    if not os.path.exists(cached_path):
        img = Image.open(image_path).convert("RGB")
        img.save(cached_path, "PNG")

    return [cached_path]


def process_text_to_images(text_path: str) -> List[str]:
    """
    将纯文本文件（.txt / .md）渲染为图像页面，使其可被 ColPali 建立视觉索引。

    渲染策略：
    - 固定画布：850×1200（近似 A4 比例），白色背景，黑色字体
    - 长行自动折行（约每行 60 字符宽），超出画布高度时自动分页
    - 字体优先使用系统字体，失败则用 Pillow 内置字体

    参数:
        text_path (str): .txt 或 .md 文件路径

    返回:
        List[str]: 渲染后各页 PNG 图片路径列表
    """
    file_hash = get_file_hash(text_path)
    doc_cache_dir = os.path.join(IMAGE_CACHE_DIR, file_hash)
    os.makedirs(doc_cache_dir, exist_ok=True)

    # 读取文本内容（尝试 UTF-8，失败则用 latin-1 兜底）
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        with open(text_path, "r", encoding="latin-1") as f:
            raw_text = f.read()

    # 清除旧缓存（防止字体/布局变更后残留破损图片）
    if os.path.exists(doc_cache_dir):
        shutil.rmtree(doc_cache_dir)
    os.makedirs(doc_cache_dir, exist_ok=True)

    font = _get_font(_TEXT_FONT_SIZE)
    usable_w = _TEXT_PAGE_W - 2 * _TEXT_MARGIN  # 可用宽度（px）

    # 估算每行可容纳的字符数（以 ASCII 宽度 ~10px 为基准，CJK 字符约为 2×）
    chars_per_line = max(20, usable_w // 10)

    # 将每行原始文本折行为若干展示行
    display_lines: List[str] = []
    for raw_line in raw_text.splitlines():
        if raw_line.strip() == "":
            display_lines.append("")  # 保留空行（段落间距）
        else:
            wrapped = textwrap.wrap(raw_line, width=chars_per_line) or [""]
            display_lines.extend(wrapped)

    # 按画布高度分页
    max_lines_per_page = (_TEXT_PAGE_H - 2 * _TEXT_MARGIN) // _TEXT_LINE_HEIGHT

    image_paths: List[str] = []
    page_num = 0
    line_idx = 0

    while line_idx < len(display_lines):
        page_num += 1
        cached_path = os.path.join(doc_cache_dir, f"page_{page_num}.png")

        img = Image.new("RGB", (_TEXT_PAGE_W, _TEXT_PAGE_H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        y = _TEXT_MARGIN
        lines_on_page = 0
        while line_idx < len(display_lines) and lines_on_page < max_lines_per_page:
            line = display_lines[line_idx]
            draw.text((_TEXT_MARGIN, y), line, fill=(30, 30, 30), font=font)
            y += _TEXT_LINE_HEIGHT
            line_idx += 1
            lines_on_page += 1

        img.save(cached_path, "PNG")

        image_paths.append(cached_path)

    return image_paths
