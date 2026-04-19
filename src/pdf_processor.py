import os
import shutil
import hashlib
from typing import List
from PIL import Image
from pdf2image import convert_from_path
from src.config import IMAGE_CACHE_DIR

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
