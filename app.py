import os
import uuid
from datetime import datetime

import streamlit as st
from PIL import Image

# 导入拆分重构好的 RAG 组件
from src.config import DEFAULT_TOP_K, IMAGE_CACHE_DIR, MAX_QUERY_CHARS
from src.pdf_processor import get_file_hash, process_pdf_to_images
from src.vector_store import VisionVectorStore
from src.llm_generator import generate_answer_with_vision

APP_BUILD_VERSION = "2026-04-19-1"

# 配置全局页面标题与布局结构，这在求职简历上可以提供更良好的用户体验
st.set_page_config(page_title="ColPali 多模态RAG - 简历级项目", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(100% 120% at 0% 0%, #f5f7fb 0%, #eef3f9 35%, #f8fafc 100%);
    }
    .hero-card {
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 20px;
        background: linear-gradient(145deg, #ffffff 0%, #f4f8ff 100%);
        padding: 18px 20px;
        box-shadow: 0 12px 36px rgba(17, 24, 39, 0.07);
        margin-bottom: 14px;
    }
    .muted-note {
        color: #4b5563;
        font-size: 0.94rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# 全局状态管理 (Session State) 与初始化
# ================================

@st.cache_resource
def get_vector_store(engine_version: str):
    """
    单例模式加载耗存模型，这对于包含大模型的 Streamlit 应用至关重要，
    避免每次用户点击网页就重新下载几 GB 的 ColPali 参数，保证体验流畅度。
    """
    _ = engine_version
    return VisionVectorStore()

# 当页面第一次启动时先初始化缓存与组件状态
if "vs" not in st.session_state:
    with st.spinner("🚀 大模型引擎冷启动：正在将本地磁盘中的 ColPali 百亿大模型载入至系统内存，首次启动需数秒，请耐心等待..."):
        st.session_state.vs = get_vector_store(APP_BUILD_VERSION)
        st.success("✅ 核心引擎已加载完毕！")

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None

if "retrieved_evidences" not in st.session_state:
    st.session_state.retrieved_evidences = []

if "generated_answer" not in st.session_state:
    st.session_state.generated_answer = ""

if "query_scope" not in st.session_state:
    st.session_state.query_scope = "当前文档"

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "chat_order" not in st.session_state:
    st.session_state.chat_order = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


def create_chat(title: str = "新对话") -> str:
    chat_id = uuid.uuid4().hex[:12]
    st.session_state.chats[chat_id] = {
        "chat_id": chat_id,
        "title": title,
        "messages": [],
        "document_ids": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.chat_order.insert(0, chat_id)
    st.session_state.current_chat_id = chat_id
    return chat_id


def ensure_active_chat() -> str:
    if st.session_state.current_chat_id in st.session_state.chats:
        return st.session_state.current_chat_id
    if st.session_state.chat_order:
        st.session_state.current_chat_id = st.session_state.chat_order[0]
        return st.session_state.current_chat_id
    return create_chat()


def get_active_chat() -> dict:
    chat_id = ensure_active_chat()
    return st.session_state.chats[chat_id]


def append_chat_message(role: str, content: str, evidences=None):
    chat = get_active_chat()
    payload = {"role": role, "content": content}
    if evidences is not None:
        payload["evidences"] = evidences
    chat["messages"].append(payload)

# ==================================
# 侧边栏: Gemini 风格聊天列表 + 文档库
# ==================================
if not st.session_state.chat_order:
    create_chat()

active_chat_id = ensure_active_chat()
active_chat = get_active_chat()

st.sidebar.markdown(
    """
    <div style="padding: 4px 0 10px 0;">
        <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">VisionRAG</div>
        <div style="color: #64748b; font-size: 0.92rem;">Gemini 风格工作台 · 左侧对话，右侧内容</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("＋ 新建聊天", use_container_width=True):
    create_chat()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("聊天记录")

for chat_id in st.session_state.chat_order:
    chat = st.session_state.chats[chat_id]
    label = chat["title"]
    if chat_id == st.session_state.current_chat_id:
        label = f"● {label}"
    if st.sidebar.button(label, key=f"chat-{chat_id}", use_container_width=True):
        st.session_state.current_chat_id = chat_id
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📁 文档知识库")
st.sidebar.markdown(
    """
    <div style="color:#475569;font-size:0.92rem;line-height:1.55;">
    支持多文档并存；默认仅检索当前文档，切换到“全部文档”时会在整个库里召回。
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.documents:
    selectable_doc_ids = list(st.session_state.documents.keys())
    default_idx = 0
    if st.session_state.current_document_id in selectable_doc_ids:
        default_idx = selectable_doc_ids.index(st.session_state.current_document_id)

    selected_doc_id = st.sidebar.selectbox(
        "当前文档",
        options=selectable_doc_ids,
        index=default_idx,
        format_func=lambda doc_id: st.session_state.documents[doc_id]["name"],
    )
    st.session_state.current_document_id = selected_doc_id

    st.sidebar.caption("已入库文档")
    for meta in st.session_state.documents.values():
        active_badge = " · 当前" if meta["document_id"] == st.session_state.current_document_id else ""
        st.sidebar.write(f"- {meta['name']} · {meta['pages']} 页{active_badge}")
else:
    st.sidebar.info("还没有文档。先上传一个 PDF 再开始对话。")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("上传 PDF", type="pdf")

if uploaded_file is not None:
    temp_pdf_path = os.path.join(IMAGE_CACHE_DIR, uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_hash = get_file_hash(temp_pdf_path)
    document_id = file_hash

    st.sidebar.success(f"已接收: {uploaded_file.name}")

    if st.sidebar.button("立即入库", use_container_width=True):
        with st.spinner("正在按页截图、计算视觉向量并写入 Qdrant..."):
            try:
                image_paths = process_pdf_to_images(temp_pdf_path, dpi=200)
                st.session_state.vs.embed_and_store_documents(
                    image_paths,
                    document_id=document_id,
                    document_name=uploaded_file.name,
                    batch_size=4,
                    replace_document=True,
                )

                st.session_state.documents[document_id] = {
                    "document_id": document_id,
                    "name": uploaded_file.name,
                    "pages": len(image_paths),
                    "image_paths": image_paths,
                }
                st.session_state.current_document_id = document_id
                active_chat["document_ids"] = sorted(set(active_chat["document_ids"] + [document_id]))

                if active_chat["title"] == "新对话":
                    active_chat["title"] = os.path.splitext(uploaded_file.name)[0][:28]

                st.session_state.retrieved_evidences = []
                st.session_state.generated_answer = ""
                st.sidebar.success(f"入库完成，共 {len(image_paths)} 页。")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"处理发生意外错误: {e}")

# ==================================
# 主视图区域: Gemini 风格对话面板
# ==================================
st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0; font-size:2.15rem; color:#0f172a;">Enterprise VisionRAG Workspace</h1>
        <p class="muted-note" style="margin:8px 0 0 0;">左侧切换聊天，右侧进行多模态检索问答。默认只看当前文档，证据页可追溯。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3 = st.columns([1.2, 1.2, 1.2])
with status_col1:
    st.metric("聊天数", len(st.session_state.chat_order))
with status_col2:
    st.metric("文档数", len(st.session_state.documents))
with status_col3:
    st.metric("当前聊天", active_chat["title"])

st.session_state.query_scope = st.radio(
    "检索范围",
    options=["当前文档", "全部文档"],
    index=0 if st.session_state.query_scope == "当前文档" else 1,
    horizontal=True,
)

for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("role") == "assistant" and message.get("evidences"):
            with st.expander("查看证据页"):
                for idx, evidence in enumerate(message["evidences"]):
                    path = evidence["image_path"]
                    page_no = evidence.get("page_number", "?")
                    doc_name = evidence.get("document_name", os.path.basename(path))
                    score = float(evidence.get("score", 0.0))
                    st.markdown(f"**{idx + 1}.** {doc_name} · 第 {page_no} 页 · 相似度 {score:.3f}")
                    img = Image.open(path)
                    st.image(img, use_column_width=True, caption=os.path.basename(path))

user_query = st.chat_input("输入你想问的问题，Gemini 风格对话就从这里开始...")

if user_query:
    if active_chat["title"] == "新对话":
        active_chat["title"] = user_query.strip()[:28]

    append_chat_message("user", user_query)

    if not st.session_state.documents:
        append_chat_message("assistant", "请先在左侧上传 PDF 并完成入库，然后再开始提问。")
        st.rerun()

    if len(user_query.strip()) == 0:
        append_chat_message("assistant", "问题内容不能为空。")
        st.rerun()

    if len(user_query) > MAX_QUERY_CHARS:
        append_chat_message("assistant", f"问题过长，请控制在 {MAX_QUERY_CHARS} 个字符以内。")
        st.rerun()

    if st.session_state.query_scope == "当前文档":
        target_document_ids = [st.session_state.current_document_id] if st.session_state.current_document_id else []
        if not target_document_ids and active_chat["document_ids"]:
            target_document_ids = [active_chat["document_ids"][-1]]
    else:
        target_document_ids = list(st.session_state.documents.keys())

    if not target_document_ids:
        append_chat_message("assistant", "当前没有可检索的文档，请先上传并入库。")
        st.rerun()

    with st.status("正在执行两阶段检索并生成答案...", expanded=True) as status:
        st.write("1. 生成查询向量并做 MUVERA 压缩表示")
        st.write(f"2. 从 Qdrant 中召回 TopK={DEFAULT_TOP_K} 页相关证据")
        results = st.session_state.vs.retrieve_with_two_stage(
            user_query,
            top_k=DEFAULT_TOP_K,
            document_ids=target_document_ids,
        )
        st.write("3. 将证据页发送给 Doubao 生成最终回答")

        retrieved_paths = [item["image_path"] for item in results if item.get("image_path")]
        st.session_state.retrieved_evidences = results
        answer = generate_answer_with_vision(user_query, retrieved_paths)
        st.session_state.generated_answer = answer
        append_chat_message("assistant", answer, evidences=results)
        status.update(label="完成", state="complete", expanded=False)

    st.rerun()
