"""文档预处理模块。

这个模块负责把用户上传的视觉文档统一整理成“页面图像列表”，供后续
ColPali 视觉检索模型使用。项目当前主打的输入类型是：
1. PDF：直接拆成逐页图片。
2. 单张图片：作为单页文档进入索引。
3. PPTX：先转成 PDF，再复用 PDF 转页图链路。

后面的向量化、检索、生成都默认依赖这种统一的页面图表示。
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import textwrap
from typing import Callable, List, Optional
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from src.config import IMAGE_CACHE_DIR

# 纯文本渲染成页面图时使用的固定画布参数。
# 这些值共同决定了文本页的视觉密度，进而影响 ColPali 能看到的版面结构。
_TEXT_PAGE_W = 850
_TEXT_PAGE_H = 1200
_TEXT_MARGIN = 60       # px，四周留白
_TEXT_FONT_SIZE = 18
_TEXT_LINE_HEIGHT = 30  # px


def _get_font(size: int = _TEXT_FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """获取适合文本渲染的字体对象。

    文本文件最终会被绘制成图片页，所以字体是否支持中文会直接影响检索质量。
    这里优先尝试系统中常见的中文字体；如果运行环境缺少这些字体，再退回到
    Pillow 自带字体，至少保证流程可运行，但中文显示效果可能会变差。
    """
    candidates = [
        # macOS 常见中文字体，优先尝试以保证本地开发时的显示质量。
        "/System/Library/Fonts/PingFang.ttc",               # 苹方（macOS 10.11+，完整 CJK）
        "/System/Library/Fonts/STHeiti Medium.ttc",          # 华文黑体中等
        "/System/Library/Fonts/STHeiti Light.ttc",           # 华文黑体细体
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux 环境下常见的中文字体。
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # 最后的兜底字体：至少能显示英文和数字，但不保证中文正常。
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    # 按候选顺序尝试，谁能成功加载就用谁，避免因为单一字体缺失导致整个流程失败。
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Pillow 10+ 的 load_default 支持 size 参数；旧版本则退回无 size 参数的调用方式。
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def clear_all_caches() -> None:
    """
    清理页面图缓存目录。

    这里只删除 `IMAGE_CACHE_DIR` 中的中间图片，不会直接修改 Qdrant 向量库。
    它的主要用途是：开发调试时需要重建缓存，或者磁盘上的旧图片已经不可信。
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
    这个哈希值在本项目里承担两个角色：
    1. 作为缓存目录名的一部分，把同一份文件稳定映射到同一批页面图。
    2. 作为 document_id 的候选来源，帮助索引层识别“这是不是同一个文档”。

    这里按文件内容而不是按文件名计算哈希，是为了避免“同内容不同文件名”
    被误判成不同文档。
    
    参数:
        file_path (str): 文件路径
        
    返回:
        str: 32位 MD5 字符串
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取可以避免一次性把整份大文件全部读入内存。
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_soffice_binary() -> str | None:
    """定位可用的 LibreOffice/soffice 可执行文件。"""
    candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def process_pdf_to_images(
    pdf_path: str,
    dpi: int = 150,
    cache_key: str | None = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """
    将 PDF 文件的每一页分别转为高质量 PNG 截图并保存到缓存目录。
    这是整个 Vision RAG 流程最关键的输入转换步骤之一。

    为什么 PDF 不直接抽文本，而要先转图片：
    1. 视觉检索模型 ColPali 本身就是“看页面图”而不是“读纯文本”。
    2. PDF 中的表格、图表、公式、双栏排版、页眉页脚等信息，在纯文本抽取时
       很容易丢失结构；而整页图片能最大程度保留这些布局线索。
    3. 后续前端展示证据页时，也需要一张可直接显示的页面图。

    DPI 选择 150 的核心原因是平衡“清晰度”和“体积/速度”：
    页面太小会导致细字不可读，页面太大又会增加缓存体积和后续处理成本。
    
    参数:
        pdf_path (str): PDF 文件的绝对/相对路径。
        dpi (int): 分辨率（默认为150）。ColPali 推理前将图像固定 resize 到 448×448，
                   300 DPI 的像素（2480×3508）完全超出模型所需，检索质量与 150 DPI 等同；
                   150 DPI（1240×1754）在前端全屏查看时仍清晰可读，同时渲染速度约快 2×、缓存体积约小 75%。
        
    返回:
        List[str]: 所有页面对应的图片保存路径列表。
    """
    file_hash = cache_key or get_file_hash(pdf_path)
    # 每个文件使用独立缓存目录，避免不同文档之间的页面图相互覆盖。
    doc_cache_dir = os.path.join(IMAGE_CACHE_DIR, file_hash)
    os.makedirs(doc_cache_dir, exist_ok=True)
    
    try:
        # `pdf2image` 会把 PDF 每一页展开成一个 PIL Image 对象。
        # 这里依赖系统安装的 poppler；在 macOS 上通常通过 `brew install poppler` 安装。
        pages = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    except Exception as e:
        raise RuntimeError(f"PDF 文档 {pdf_path} 读取失败，请检查是否为受损文件或系统是否安装了 poppler. 错误信息: {str(e)}")

    image_paths = []
    # 逐页落盘。若本地已经存在对应缓存，则直接复用，避免重复渲染 PDF。
    for i, page in enumerate(pages):
        image_name = f"page_{i + 1}.png"
        image_path = os.path.join(doc_cache_dir, image_name)
        
        if not os.path.exists(image_path):
            page.save(image_path, "PNG")
            
        # 后续向量化和前端证据展示都依赖这条物理路径。
        image_paths.append(image_path)
        if on_progress:
            on_progress(i + 1, len(pages))
        
    return image_paths


def process_pptx_to_images(
    pptx_path: str,
    dpi: int = 150,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """将 PPTX 先转成 PDF，再复用现有 PDF 渲染链路生成页面图。"""
    soffice_binary = _find_soffice_binary()
    if not soffice_binary:
        raise RuntimeError(
            "未检测到 LibreOffice/soffice，无法处理 PPTX。"
            "请先安装 LibreOffice，并确保 `soffice` 可执行文件可用。"
        )

    cache_key = get_file_hash(pptx_path)
    with tempfile.TemporaryDirectory(prefix="rag_pptx_convert_") as output_dir:
        convert_cmd = [
            soffice_binary,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            pptx_path,
        ]
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "未知错误").strip()
            raise RuntimeError(f"PPTX 转 PDF 失败：{error_message}")

        pdf_name = f"{os.path.splitext(os.path.basename(pptx_path))[0]}.pdf"
        converted_pdf_path = os.path.join(output_dir, pdf_name)
        if not os.path.exists(converted_pdf_path):
            raise RuntimeError("PPTX 转 PDF 失败：未找到转换后的 PDF 文件")

        return process_pdf_to_images(
            converted_pdf_path,
            dpi=dpi,
            cache_key=cache_key,
            on_progress=on_progress,
        )


def process_image_to_images(
    image_path: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """
    处理单张图片文件（PNG/JPG/JPEG/WEBP），将其作为"单页文档"纳入索引。
    这条路径比 PDF 简单，因为输入本来就是图片，但仍然要做一次格式标准化：
    后续检索链路希望所有页面都以统一的 PNG 文件形式存在，减少不同图片格式
    在颜色模式、透明通道、解码行为上的差异。
    
    参数:
        image_path (str): 图片文件路径
        
    返回:
        List[str]: 包含缓存后图片路径的单元素列表
    """
    file_hash = get_file_hash(image_path)
    doc_cache_dir = os.path.join(IMAGE_CACHE_DIR, file_hash)
    os.makedirs(doc_cache_dir, exist_ok=True)

    # 统一缓存成 RGB PNG，避免 JPG / WEBP / 调色板图在后续处理时出现颜色模式不一致。
    cached_path = os.path.join(doc_cache_dir, "page_1.png")

    if not os.path.exists(cached_path):
        img = Image.open(image_path).convert("RGB")
        img.save(cached_path, "PNG")

    if on_progress:
        on_progress(1, 1)

    return [cached_path]


def process_text_to_images(
    text_path: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """
    将纯文本文件（.txt / .md）渲染为图像页面，使其可被 ColPali 建立视觉索引。

    这个函数目前主要保留给历史数据兼容或离线实验使用；默认上传入口已不再
    暴露 TXT / MD。

    这一点很重要：虽然输入是文本，但检索模型不是传统文本 embedding 模型，
    而是“看页面截图”的视觉模型。所以文本文件不能直接跳过图像化步骤，
    必须先排版成一页页白底黑字的图片，才能和 PDF 页面、原始图片走同一条链路。

    渲染策略：
    - 固定画布：850×1200（近似 A4 比例），白色背景，黑色字体
    - 长行自动折行，超出画布高度时自动分页
    - 字体优先使用系统字体，失败则用 Pillow 内置字体

    参数:
        text_path (str): .txt 或 .md 文件路径

    返回:
        List[str]: 渲染后各页 PNG 图片路径列表
    """
    file_hash = get_file_hash(text_path)
    doc_cache_dir = os.path.join(IMAGE_CACHE_DIR, file_hash)
    os.makedirs(doc_cache_dir, exist_ok=True)

    # 优先按 UTF-8 读取，这是现代文本文件最常见的编码；若失败再做兼容性兜底。
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        with open(text_path, "r", encoding="latin-1") as f:
            raw_text = f.read()

    # 文本页缓存每次都重建。
    # 原因是文本渲染高度依赖字体、行宽、页高等布局参数，一旦这些参数调整，
    # 旧缓存就不再可信，继续复用反而会造成学习和调试上的混淆。
    if os.path.exists(doc_cache_dir):
        shutil.rmtree(doc_cache_dir)
    os.makedirs(doc_cache_dir, exist_ok=True)

    font = _get_font(_TEXT_FONT_SIZE)
    usable_w = _TEXT_PAGE_W - 2 * _TEXT_MARGIN  # 文字真正可用的横向空间。

    # 这是一个近似值，不做逐字符精确测量，而是用经验宽度快速估算每行容量。
    # 这样做的目标不是排版出版级精度，而是稳定地把文本切成适合视觉模型读取的页面。
    chars_per_line = max(20, usable_w // 10)

    # 先按“原始逻辑行”处理，再把过长的行拆成多条展示行。
    # 空行会被保留，目的是让段落结构在图片中也能体现出来。
    display_lines: List[str] = []
    for raw_line in raw_text.splitlines():
        if raw_line.strip() == "":
            display_lines.append("")  # 保留空行（段落间距）
        else:
            wrapped = textwrap.wrap(raw_line, width=chars_per_line) or [""]
            display_lines.extend(wrapped)

    # 根据页面高度计算一页最多能容纳多少条展示行。
    max_lines_per_page = (_TEXT_PAGE_H - 2 * _TEXT_MARGIN) // _TEXT_LINE_HEIGHT
    total_pages = max(1, (len(display_lines) + max_lines_per_page - 1) // max_lines_per_page)

    image_paths: List[str] = []
    page_num = 0
    line_idx = 0

    # 逐页绘制，直到所有展示行都被消耗完。
    while line_idx < len(display_lines):
        page_num += 1
        cached_path = os.path.join(doc_cache_dir, f"page_{page_num}.png")

        img = Image.new("RGB", (_TEXT_PAGE_W, _TEXT_PAGE_H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        y = _TEXT_MARGIN
        lines_on_page = 0
        # 当前页从上到下逐行绘制，画满一页后继续创建下一页。
        while line_idx < len(display_lines) and lines_on_page < max_lines_per_page:
            line = display_lines[line_idx]
            draw.text((_TEXT_MARGIN, y), line, fill=(30, 30, 30), font=font)
            y += _TEXT_LINE_HEIGHT
            line_idx += 1
            lines_on_page += 1

        img.save(cached_path, "PNG")

        image_paths.append(cached_path)
        if on_progress:
            on_progress(page_num, total_pages)

    return image_paths
