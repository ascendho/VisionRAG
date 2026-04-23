import base64
from typing import List
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


def generate_answer_with_vision(query_text: str, image_paths: List[str], max_tokens: int = 600) -> str:
    """
    基于火山引擎 Doubao-seed-2.0-pro 能够处理多模态图像的模型给出最终的 RAG 总结。

    返回：
        answer_text: Markdown 格式的回答字符串
    """

    if not ARK_API_KEY:
        return "无法连接大模型服务：请检查 `.env` 文件中是否设置了有效的 `ARK_API_KEY`。"

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
    )

    system_prompt = (
        "你是一个精准的 AI 文档助手。请仔细阅读以下文档截图，"
        "用流利的中文回答用户的问题，使用结构化 Markdown 格式输出。"
    )

    content_list: List[dict] = [{"type": "text", "text": system_prompt}]

    for image_path in image_paths[:10]:
        img = Image.open(image_path)
        base64_img = pil_image_to_base64(img)
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_img}"}
        })

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
