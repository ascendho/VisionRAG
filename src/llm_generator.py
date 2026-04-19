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

def generate_answer_with_vision(query_text: str, image_paths: List[str], max_tokens: int = 500) -> str:
    """
    基于火山引擎 Doubao-seed-2.0-pro 能够处理多模态图像的模型给出最终的 RAG 总结。
    
    参数:
        query_text (str): 用户的问题/指令
        image_paths (List[str]): 经检索系统定位到的最相关的数张 PDF 截图路径。
        max_tokens (int): 最大输出长度。
        
    返回:
        str: 模型的回答。
    """
    
    if not ARK_API_KEY:
        return "无法连接大模型服务：请检查 `.env` 文件中是否设置了有效的 `ARK_API_KEY`。"
    
    # 按照字节跳动火山引擎的官方模板初始化
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
    )
    
    # 构建用户提问的具体指令要求
    content_list = [
        {
            "type": "text",
            "text": "系统指令：你是一个非常有帮助的 AI 文档助手。请仔细阅读以下给你的几张文档截图，用流利的中文回答用户的问题。输出必须具有结构化且使用 Markdown 格式。\n\n"
        }
    ]

    # 加入每一张排名前列的高质量截图数据以作证
    for image_path in image_paths[:10]:
        img = Image.open(image_path)
        base64_img = pil_image_to_base64(img)

        # 火山多模态接收 Base64 的特定 schema: data协议
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}"
            }
        })
        
    # 追加具体的用户提问
    content_list.append({
        "type": "text",
        "text": f"用户提问：{query_text}"
    })

    try:
        # 执行火云 API 请求并返回
        # 备注：Volcengine openai-python package wrapper 通常暴露在 chat.completions，也可兼容自定义的结构。
        response = client.chat.completions.create(
            model=DOUBAO_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
        
    except AttributeError:
        # 如果是基于部分极早期的官方扩展包，使用 responses.create
        try:
            response = client.responses.create(
                model=DOUBAO_MODEL_NAME,
                input=[
                    {
                        "role": "user",
                        "content": content_list
                    }
                ]
            )
            # 根据 responses.create 的返回值进行结构尝试拉取
            return str(response) if not hasattr(response, "choices") else response.choices[0].message.content
        except Exception as sub_e:
            return f"模型双路推理生成响应异常。详情: {sub_e}"
    except Exception as e:
        return f"模型推理生成响应异常。网络或配置可能存在问题。详情: {e}"
