import base64
import json
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


def _build_system_prompt() -> str:
    return (
        "你是一个谨慎、可追溯的 AI 文档助手。"
        "你只能依据当前回合提供的文档截图、证据元数据以及必要的对话历史回答问题，"
        "不能使用证据之外的常识、训练记忆或猜测补全事实。\n"
        "回答规则：\n"
        "1. 默认使用简体中文，并用清晰的 Markdown 输出。\n"
        "2. 先直接回答问题，再补充依据；如果无法直接回答，要明确说明原因。\n"
        "3. 提到具体事实、数字、时间、名称、步骤或结论时，必须在句末附上来源，格式为 [E1]、[E2] 这样的证据编号。\n"
        "4. 如果当前截图无法读清、证据不足、证据相互冲突，必须明确说明“根据当前证据无法确认”，不要编造。\n"
        "5. 如果用户问题超出当前证据范围，要直接说明超出范围，不要借助常识扩写。\n"
        "6. 如果历史对话与当前证据冲突，以当前证据为准，并指出冲突点。\n"
        "7. 不要声称看过未提供的页面，也不要编造证据编号、页码、文档名或引用。\n"
        "8. 输出尽量使用以下结构中的适用部分：`结论`、`依据`、`不确定点`。"
    )


def _format_evidence_context(evidence_context: List[dict] | None = None) -> str:
    if not evidence_context:
        return ""

    lines = ["本轮可用证据页面如下。只有这些证据编号允许在回答中被引用："]
    for index, item in enumerate(evidence_context, start=1):
        evidence_id = item.get("evidence_id", f"E{index}")
        document_name = item.get("document_name", "未知文档")
        page_number = item.get("page_number", "?")
        score = item.get("score")
        if isinstance(score, (float, int)):
            lines.append(f"[{evidence_id}] 文档：{document_name}；页码：{page_number}；相关性分数：{score:.2f}")
        else:
            lines.append(f"[{evidence_id}] 文档：{document_name}；页码：{page_number}")
    return "\n".join(lines)


def _build_turn_sections(
    query_text: str,
    evidence_context: List[dict] | None = None,
) -> List[str]:
    sections = [
        "请只根据当前提供的文档截图和证据元数据回答下面的问题。",
        "如果证据不足、图片不可读或信息冲突，请直接说明，不要猜测。",
        "如果需要引用依据，只能使用给定的证据编号，格式必须是 [E1]、[E2]。",
    ]
    evidence_text = _format_evidence_context(evidence_context)
    if evidence_text:
        sections.append(evidence_text)
    sections.append(f"用户问题：{query_text}")
    return sections


def _build_current_turn_content(
    query_text: str,
    image_paths: List[str],
    evidence_context: List[dict] | None = None,
) -> List[dict]:
    turn_sections = _build_turn_sections(query_text, evidence_context)

    content_list: List[dict] = [{"type": "text", "text": "\n\n".join(turn_sections)}]
    content_list.extend(_build_image_content_list(image_paths))
    return content_list


def _parse_suggested_questions(raw_text: str, max_questions: int) -> List[str]:
    cleaned = raw_text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            questions = [str(item).strip() for item in parsed if str(item).strip()]
            return questions[:max_questions]
    except Exception:
        pass

    questions = []
    for line in cleaned.splitlines():
        candidate = line.strip().lstrip("-•0123456789. ").strip()
        if candidate:
            questions.append(candidate)
    return questions[:max_questions]


def generate_suggested_questions(
    document_name: str,
    image_paths: List[str],
    evidence_context: List[dict] | None = None,
    max_questions: int = 4,
) -> List[str]:
    if not ARK_API_KEY or not image_paths:
        return []

    client = _build_client()
    system_prompt = (
        "你是一个文档问答助手。"
        "你只能根据当前提供的文档截图生成后续可直接提问的中文问题。"
        "每个问题都必须能从这些页面中得到回答，不要生成依赖外部知识的问题。"
        f"请输出 {max_questions} 个高质量问题，只返回 JSON 字符串数组。"
    )

    turn_sections = [
        f"文档名称：{document_name}",
        f"请生成 {max_questions} 个适合用户继续提问的问题。",
        "问题应覆盖内容总结、关键数据、流程步骤、限制条件或适用场景等不同角度。",
        "不要重复，不要输出答案，只返回问题列表。",
    ]
    evidence_text = _format_evidence_context(evidence_context)
    if evidence_text:
        turn_sections.append(evidence_text)

    content_list: List[dict] = [{"type": "text", "text": "\n\n".join(turn_sections)}]
    content_list.extend(_build_image_content_list(image_paths))

    try:
        response = client.chat.completions.create(
            model=DOUBAO_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ],
            max_tokens=300,
        )
        raw_text = response.choices[0].message.content.strip()
        return _parse_suggested_questions(raw_text, max_questions)
    except Exception:
        return []


def generate_answer_with_vision(
    query_text: str,
    image_paths: List[str],
    evidence_context: List[dict] | None = None,
    max_tokens: int = 600,
) -> str:
    """
    基于火山引擎 Doubao-seed-2.0-pro 能够处理多模态图像的模型给出最终的 RAG 总结。

    返回：
        answer_text: Markdown 格式的回答字符串
    """

    if not ARK_API_KEY:
        return "无法连接大模型服务：请检查 `.env` 文件中是否设置了有效的 `ARK_API_KEY`。"

    client = _build_client()
    system_prompt = _build_system_prompt()
    content_list = _build_current_turn_content(query_text, image_paths, evidence_context)

    try:
        response = client.chat.completions.create(
            model=DOUBAO_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except AttributeError:
        try:
            response = client.responses.create(
                model=DOUBAO_MODEL_NAME,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_list},
                ]
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
    evidence_context: List[dict] | None = None,
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
    stream = None

    system_message = {
        "role": "system",
        "content": _build_system_prompt(),
    }

    # 当前轮：图像 + 问题
    current_content = _build_current_turn_content(query_text, image_paths, evidence_context)

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
    finally:
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            try:
                close_stream()
            except Exception:
                pass
