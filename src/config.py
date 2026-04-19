import os
import tempfile
from dotenv import load_dotenv

# 加载当前目录下的 .env 文件中定义的环境变量
load_dotenv()

# Qdrant 数据库连接地址（本地 Docker 环境运行 Qdrant）
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# 火山引擎相关参数
ARK_API_KEY = os.getenv("ARK_API_KEY")
DOUBAO_MODEL_NAME = "doubao-seed-2-0-pro-260215"

# 向量数据库的集合（Collection）名称
COLLECTION_NAME = "colpali-rag-collection"

# 检索与输入控制参数
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "800"))

# 临时存放 PDF 生成图片的目录，避免每次都转换。
# 放在系统临时目录，避免触发 uvicorn --reload 对项目目录变更的热重启。
IMAGE_CACHE_DIR = os.getenv(
	"IMAGE_CACHE_DIR",
	os.path.join(tempfile.gettempdir(), "rag_cache_images")
)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

# ColPali 视觉语言模型名称
# 使用 vidore/colpali-v1.3 为文档和查询生成多模态特征向量（Embeddings）
COLPALI_MODEL_NAME = "vidore/colpali-v1.3"
