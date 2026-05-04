"""项目核心配置。

这个文件是整条 RAG 链路的最低层依赖，负责集中定义：
1. 外部服务地址，例如 Qdrant 和 Doubao API。
2. 检索与输入约束，例如 Top-K、最低分阈值、查询长度上限。
3. 中间产物目录，例如文档页图片缓存目录。

之所以把这些配置放在一个文件里，而不是散落在各个模块中，是为了让
`doc_processor.py`、`vector_store.py`、`llm_generator.py` 在读取运行参数时
都共享同一套事实来源，避免出现不同模块各自维护默认值的情况。
"""

import os
import shutil
import tempfile
from dotenv import load_dotenv

# 在模块导入阶段就读取 `.env`，这样后续所有模块拿到的都是统一配置。
load_dotenv()

# ColPali 权重通常从 Hugging Face 下载。
# 这里允许通过国内镜像源减少网络不稳定带来的下载失败，尤其适合本地开发环境。
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def _resolve_runtime_dir(env_key: str, default_relative_path: str, *, legacy_relative_path: str | None = None) -> str:
	configured_path = os.getenv(env_key)
	resolved_path = os.path.abspath(configured_path) if configured_path else os.path.join(PROJECT_ROOT, default_relative_path)

	if legacy_relative_path and not configured_path:
		legacy_path = os.path.join(PROJECT_ROOT, legacy_relative_path)
		if os.path.exists(legacy_path) and not os.path.exists(resolved_path):
			os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
			shutil.move(legacy_path, resolved_path)

	os.makedirs(resolved_path, exist_ok=True)
	return resolved_path


DATA_DIR = _resolve_runtime_dir("DATA_DIR", "data")

# Qdrant 是本项目的向量数据库，用来保存每一页文档的多向量表示并执行检索。
# 默认指向本地 Docker 容器暴露的 6333 端口。
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# 在本地开发环境里，如果 Docker 版 Qdrant 不可用，就回退到仓库内的持久化目录。
QDRANT_PATH = _resolve_runtime_dir("QDRANT_PATH", "data/qdrant", legacy_relative_path="qdrant_local")

# 上传的原始文件和 embedded Qdrant 数据一起落在同一片运行数据目录里，方便统一清理与迁移。
UPLOADED_FILES_DIR = _resolve_runtime_dir(
	"UPLOADED_FILES_DIR",
	os.path.join("data", "qdrant", "pdfs"),
)

# benchmark 运行结果默认保存在 data/eval_runs，避免仓库根目录持续堆积临时产物。
EVAL_RUNS_DIR = _resolve_runtime_dir("EVAL_RUNS_DIR", os.path.join("data", "eval_runs"), legacy_relative_path="eval_runs")

# Doubao 是最终负责“看证据页并组织答案”的多模态大模型。
# 这里保留 API Key 和模型名两个最核心的外部调用参数。
ARK_API_KEY = os.getenv("ARK_API_KEY")
DOUBAO_MODEL_NAME = os.getenv("DOUBAO_MODEL_NAME", "doubao-seed-2-0-pro-260215")

# 所有文档页向量都落到同一个集合中，再通过 payload 中的 document_id 做过滤。
COLLECTION_NAME = "colpali-rag-collection"

# 这些参数共同决定了“用户问题如何进入后端”以及“检索结果如何被截断”。
# - DEFAULT_TOP_K: 默认最多把多少页候选证据送入后续流程。
# - DEFAULT_MIN_SCORE: 低于该分数的页面默认视为不够相关。
# - MAX_QUERY_CHARS: 防止异常超长输入把检索和生成链路拖慢。
# - QUERY_GUARD_ENABLED: 是否启用对明显闲聊/越界问题的前置拦截。
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.6"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "800"))
QUERY_GUARD_ENABLED = os.getenv("QUERY_GUARD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

# 多轮检索改写配置。
# 这组参数只影响“送去检索的查询”，不会改写用户在聊天界面里看到的原始问题。
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
QUERY_REWRITE_MODEL_NAME = os.getenv("QUERY_REWRITE_MODEL_NAME", "ep-m-20260411093114-9hftc")
QUERY_REWRITE_MAX_HISTORY_MESSAGES = int(os.getenv("QUERY_REWRITE_MAX_HISTORY_MESSAGES", "4"))
QUERY_REWRITE_TIMEOUT_MS = int(os.getenv("QUERY_REWRITE_TIMEOUT_MS", "2500"))
QUERY_REWRITE_TRIGGER_MAX_CHARS = int(os.getenv("QUERY_REWRITE_TRIGGER_MAX_CHARS", "48"))

# 文档页图片缓存目录。
# 所有 PDF / 图片 / 文本在进入向量化之前，都会先被标准化为页面图片并缓存在这里。
# 之所以放到系统临时目录，而不是项目目录，是为了避免 `uvicorn --reload`
# 监听到大量缓存文件变更后频繁热重启。
IMAGE_CACHE_DIR = os.getenv(
	"IMAGE_CACHE_DIR",
	os.path.join(tempfile.gettempdir(), "rag_cache_images")
)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# ColPali 是检索侧的视觉-语言模型。
# 它不负责生成自然语言答案，而是把“文档页图片”和“用户问题”编码成可检索的多向量表示。
COLPALI_MODEL_NAME = "vidore/colpali-v1.3"
