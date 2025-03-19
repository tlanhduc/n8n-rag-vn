import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Application settings
APP_NAME = "Vietnamese RAG"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
API_PREFIX = "/api"

# Model settings
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
MAX_TOKEN_LIMIT = 128
DEFAULT_CHUNK_SIZE = 110  # Safe margin below MAX_TOKEN_LIMIT
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_TOP_K = 5

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Cache settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() in ("true", "1", "t")
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000")) 