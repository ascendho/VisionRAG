"""大模型生成模块。

这个文件承接检索阶段的输出，把“用户问题 + 证据页图片 + 证据元数据”组织成
多模态请求，再交给 Doubao 模型生成最终答案或推荐问题。

从职责上看，它不负责决定“检索出哪些页”，而负责回答下面这些问题：
1. 页面图像如何转成视觉模型 API 能接收的格式。
2. 系统提示词如何限制模型只能依据当前证据作答。
3. 多轮对话历史、证据元数据、子问题拆分信息如何拼成当前轮输入。
4. 结果是一次性返回，还是以流式逐 token 返回给前端。
"""

import base64
import json
from io import BytesIO
from typing import Generator, List

from openai import OpenAI
from PIL import Image

from src.config import ARK_API_KEY, DOUBAO_MODEL_NAME


def pil_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """把 PIL 图片编码成 Base64 字符串。

    视觉模型 API 通常不能直接接收本地文件路径，因此这里先把图片内容转成
    `data:image/...;base64,...` 需要的核心字节串，后续再封装成多模态消息体。
    """
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _build_client() -> OpenAI:
    """构建 OpenAI 兼容客户端。

    虽然底层服务是火山引擎 ARK / Doubao，但它暴露的是 OpenAI 兼容接口，
    因此这里可以直接使用 OpenAI Python SDK 发起请求。
    """
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
    )


def _build_image_content_list(image_paths: List[str]) -> List[dict]:
    """把页面图片路径列表转换成模型请求中的 image_url 片段。

    当前实现最多携带前 10 张页面图，这是一个经验型上限：
    证据页太少可能不够回答问题，太多又会显著增加请求体体积和模型处理负担。
    """
    content = []
    for image_path in image_paths[:10]:
        # 这里直接读取落盘后的证据页图片，并转成 data URL 形式交给多模态模型。
        img = Image.open(image_path)
        b64 = pil_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    return content


def _build_system_prompt() -> str:
    """构建系统提示词。

    这是整个生成阶段的“行为护栏”。它的目标不是让模型更会发挥，
    而是尽可能约束模型：
    - 只能依据当前证据页和证据元数据回答。
    - 关键事实必须显式引用 [E1]、[E2] 这类证据编号。
    - 证据不足时必须承认不知道，而不是补全猜测。
    - 多子问题场景下不能漏答。
    """
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
        "8. 输出尽量使用以下结构中的适用部分：`结论`、`依据`、`不确定点`。\n"
        "9. 如果用户问题包含多个独立子问题，必须逐项回答每一项，不要只回答其中一部分。\n"
        "10. 对证据不足的子问题，要单独写明“根据当前证据无法确认”，不要让一个子问题的证据结论替代另一个子问题。"
    )


def _format_evidence_context(evidence_context: List[dict] | None = None) -> str:
    """把证据元数据转换成提示词中的纯文本上下文。

    这里不会直接放入图片内容，而是告诉模型：
    当前证据编号对应哪份文档、哪一页、相关性分数大致如何，以及它更可能对应哪个子问题。
    这样模型在回答时既“看得到图”，也“知道该怎么引用图”。
    """
    if not evidence_context:
        return ""

    lines = ["本轮可用证据页面如下。只有这些证据编号允许在回答中被引用："]
    for index, item in enumerate(evidence_context, start=1):
        evidence_id = item.get("evidence_id", f"E{index}")
        document_name = item.get("document_name", "未知文档")
        page_number = item.get("page_number", "?")
        score = item.get("score")
        direct_supported_sub_queries = item.get("direct_supported_sub_queries") or item.get("matched_sub_queries") or []
        reused_supported_sub_queries = item.get("reused_supported_sub_queries") or []
        if isinstance(score, (float, int)):
            line = f"[{evidence_id}] 文档：{document_name}；页码：{page_number}；相关性分数：{score:.2f}"
        else:
            line = f"[{evidence_id}] 文档：{document_name}；页码：{page_number}"

        if direct_supported_sub_queries:
            line += f"；直接支持：{' / '.join(direct_supported_sub_queries)}"
        if reused_supported_sub_queries:
            line += f"；可复用支持：{' / '.join(reused_supported_sub_queries)}"

        lines.append(line)
    return "\n".join(lines)


def _build_turn_sections(
    query_text: str,
    evidence_context: List[dict] | None = None,
    sub_queries: List[str] | None = None,
    unsupported_sub_queries: List[str] | None = None,
) -> List[str]:
    """构造当前轮的文本部分。

    这一步只拼文字说明，不拼图片。它负责明确告诉模型：
    - 本轮任务是什么。
    - 证据不足时该如何处理。
    - 如果识别到了多个独立子问题，应该如何分项回答。
    """
    sections = [
        "请只根据当前提供的文档截图和证据元数据回答下面的问题。",
        "如果证据不足、图片不可读或信息冲突，请直接说明，不要猜测。",
        "如果需要引用依据，只能使用给定的证据编号，格式必须是 [E1]、[E2]。",
    ]
    if sub_queries and len(sub_queries) > 1:
        sections.append("本轮识别到多个独立子问题，请逐项回答，不能遗漏任何一项。")
        sections.append("请按“子问题 1 / 子问题 2 / ...”的结构组织答案，并对每个子问题单独判断证据是否充分。")
        sections.append("识别到的子问题如下：")
        sections.extend([f"子问题 {index}: {sub_query}" for index, sub_query in enumerate(sub_queries, start=1)])
    if unsupported_sub_queries:
        sections.append("以下子问题当前仍然缺少可用的正式证据支持：")
        sections.extend([f"- {sub_query}" for sub_query in unsupported_sub_queries])
        sections.append("对这些子问题必须逐项写“根据当前证据无法确认”，不要借用其他子问题的证据替代回答。")

    evidence_text = _format_evidence_context(evidence_context)
    if evidence_text:
        sections.append(evidence_text)
    sections.append(f"用户原始问题：{query_text}" if sub_queries and len(sub_queries) > 1 else f"用户问题：{query_text}")
    return sections


def _build_current_turn_content(
    query_text: str,
    image_paths: List[str],
    evidence_context: List[dict] | None = None,
    sub_queries: List[str] | None = None,
    unsupported_sub_queries: List[str] | None = None,
) -> List[dict]:
    """把当前轮用户输入拼成多模态 content 列表。

    最终结构由两部分组成：
    1. 一段 text，说明问题、规则和证据元数据。
    2. 若干张 image_url，对应真正给模型看的证据页图片。
    """
    turn_sections = _build_turn_sections(query_text, evidence_context, sub_queries, unsupported_sub_queries)

    content_list: List[dict] = [{"type": "text", "text": "\n\n".join(turn_sections)}]
    content_list.extend(_build_image_content_list(image_paths))
    return content_list


def _parse_suggested_questions(raw_text: str, max_questions: int) -> List[str]:
    """解析模型返回的建议问题列表。

    理想情况下，模型会严格返回 JSON 数组；但真实线上调用中仍可能出现
    换行列表、带序号文本等非严格格式，所以这里做了两层解析兜底。
    """
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
    """基于一批代表页生成后续可追问的问题。

    这个函数主要服务前端“建议问题”能力：当用户刚上传完文档，还不知道该问什么时，
    让模型先看几张代表页，主动给出几个高概率能答出来的问题。
    """
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
        # 这里走非流式调用，因为建议问题通常量小、结构固定，更适合一次性拿完整结果。
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
    sub_queries: List[str] | None = None,
    unsupported_sub_queries: List[str] | None = None,
    max_tokens: int = 600,
) -> str:
    """
    使用多模态模型一次性生成完整答案。

    这是非流式版本，适合需要直接拿完整回答的场景。与流式版本相比，
    它在用户体验上没有“边生成边显示”的过程，但调用逻辑更直接。

    返回：
        answer_text: Markdown 格式的回答字符串
    """

    if not ARK_API_KEY:
        return "无法连接大模型服务：请检查 `.env` 文件中是否设置了有效的 `ARK_API_KEY`。"

    # 当前轮内容由“规则文本 + 证据元数据 + 证据图片”共同组成。
    client = _build_client()
    system_prompt = _build_system_prompt()
    content_list = _build_current_turn_content(query_text, image_paths, evidence_context, sub_queries, unsupported_sub_queries)

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
    sub_queries: List[str] | None = None,
    unsupported_sub_queries: List[str] | None = None,
    max_tokens: int = 800,
) -> Generator[str, None, None]:
    """
    流式生成答案。

    这是前端聊天主路径使用的版本。它不会等待模型一次性生成完全部文本，
    而是把模型返回的 token 片段逐步 `yield` 给上层 SSE 封装代码，
    从而实现“边生成边显示”。

    `chat_history` 格式为：
    [{"role": "user"|"assistant", "content": "..."}]

    这里只保留最近 10 轮对话，是为了在保留必要上下文的同时，避免历史过长
    导致上下文窗口膨胀、延迟上升或 token 超限。
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

    # 当前轮用户输入始终由“问题文本 + 当前证据页图片”组成。
    current_content = _build_current_turn_content(query_text, image_paths, evidence_context, sub_queries, unsupported_sub_queries)

    # 消息顺序非常重要：系统规则在最前，历史对话在中间，当前轮内容放最后。
    history = (chat_history or [])[-20:]
    messages = [system_message] + history + [{"role": "user", "content": current_content}]

    try:
        stream = client.chat.completions.create(
            model=DOUBAO_MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        # OpenAI 兼容流式接口会持续返回 chunk；这里逐片抽取文本增量并交给上层。
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n⚠️ 模型推理异常：{e}"
    finally:
        # 某些 SDK/服务端实现支持显式关闭流对象；这里在 finally 中兜底回收。
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            try:
                close_stream()
            except Exception:
                pass
