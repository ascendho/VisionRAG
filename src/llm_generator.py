import base64
from typing import List, Generator
from openai import OpenAI
from PIL import Image
from io import BytesIO

from src.config import ARK_API_KEY, DOUBAO_MODEL_NAME

def pil_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """
    将 PIL 图片对象以 Base64 编码打包，为发送给视觉大模型 API 做的必需准备。
    """
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _build_client() -> OpenAI:
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
    )


def _build_image_content_list(image_paths: List[str]) -> List[dict]:
    """将页面图像列表转换为多模态 content list（最多 10 张）。"""
    content = []
    for image_path in image_paths[:10]:
        img = Image.open(image_path)
        b64 = pil_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    return content


def generate_answer_with_vision(query_text: str, image_paths: List[str], max_tokens: int = 600) -> str:
    """
    基于火山引擎 Doubao-seed-2.0-pro 能够处理多模态图像的模型给出最终的 RAG 总结。

    返回：
        answer_text: Markdown 格式的回答字符串
    """

    if not ARK_API_KEY:
        return "无法连接大模型服务：请检查 `.env` 文件中是否设置了有效的 `ARK_API_KEY`。"

    client = _build_client()

    system_prompt = (
        "你是一个精准的 AI 文档助手。请仔细阅读以下文档截图，"
        "用流利的中文回答用户的问题，使用结构化 Markdown 格式输出。"
    )

    content_list: List[dict] = [{"type": "text", "text": system_prompt}]
    content_list.extend(_build_image_content_list(image_paths))
    content_list.append({"type": "text", "text": f"用户提问：{query_text}"})

    try:
        response = client.chat.completions.create(
            model=DOUBAO_MODEL_NAME,
            messages=[{"role": "user", "content": content_list}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except AttributeError:
        try:
            response = client.responses.create(
                model=DOUBAO_MODEL_NAME,
                input=[{"role": "user", "content": content_list}]
            )
            raw = str(response) if not hasattr(response, "choices") else response.choices[0].message.content
            return raw.strip()
        except Exception as sub_e:
            return f"模型双路推理生成响应异常。详情: {sub_e}"
    except Exception as e:
        return f"模型推理生成响应异常。网络或配置可能存在问题。详情: {e}"


def generate_answer_stream(
    query_text: str,
    image_paths: List[str],
    chat_history: List[dict] | None = None,
    max_tokens: int = 800,
) -> Generator[str, None, None]:
    """
    流式生成答案。接受多轮对话历史 (chat_history)，通过 SSE 逐 token yield。

    chat_history 格式: [{"role": "user"|"assistant", "content": "..."}]
    最多保留最近 10 轮（20 条消息），防止 token 超限。
    """

    if not ARK_API_KEY:
        yield "无法连接大模型服务：请检查 ARK_API_KEY 配置。"
        return

    client = _build_client()

    system_message = {
        "role": "system",
        "content": (
            "你是一个精准的 AI 文档助手。请仔细阅读以下文档截图，"
            "用流利的中文回答用户的问题，使用结构化 Markdown 格式输出。"
        )
    }

    # 当前轮：图像 + 问题
    current_content: List[dict] = []
    current_content.extend(_build_image_content_list(image_paths))
    current_content.append({"type": "text", "text": f"用户提问：{query_text}"})

    # 构建消息列表：系统 → 历史（最近 10 轮 = 20 条）→ 当前
    history = (chat_history or [])[-20:]
    messages = [system_message] + history + [{"role": "user", "content": current_content}]

    try:
        stream = client.chat.completions.create(
            model=DOUBAO_MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n⚠️ 模型推理异常：{e}"
