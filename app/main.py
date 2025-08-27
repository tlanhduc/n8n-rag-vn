import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.services.preprocessor_simple import SimpleTextPreprocessor
from app.services.embeddings_simple import SimpleEmbeddingService
from app.config import APP_NAME, API_PREFIX, DEBUG, HOST, PORT

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize services
preprocessor = SimpleTextPreprocessor()
embedding_service = SimpleEmbeddingService()

# Create FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="API for Vietnamese RAG (Retrieval Augmented Generation)",
    version="1.0.0",
    debug=DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Root endpoint
@app.get("/")
async def root():
    return {
        "app": APP_NAME,
        "version": "1.0.0",
        "status": "active",
        "api_docs": "/docs",
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# API Status endpoint
@app.get(f"{API_PREFIX}/status")
async def get_status():
    """Get server status and configuration."""
    return {
        "service": APP_NAME,
        "status": "running",
        "debug": DEBUG,
        "embedding_model": embedding_service.model_name if embedding_service.model else "fallback",
        "preprocessor": "simple",
        "cache_enabled": embedding_service.cache is not None
    }

# Test endpoint
@app.get(f"{API_PREFIX}/test")
async def test_endpoint():
    """Test endpoint to verify the application is working."""
    try:
        # Test preprocessor
        test_text = "Đây là một văn bản tiếng Việt để test. Nó có nhiều câu và từ khác nhau."
        chunks = preprocessor.create_chunks(test_text, chunk_size=20, chunk_overlap=5)
        
        # Test embedding service
        test_embedding = embedding_service.get_embedding("test text")
        
        return {
            "status": "success",
            "preprocessor_test": {
                "input_text": test_text,
                "chunks_created": len(chunks),
                "chunks": chunks
            },
            "embedding_test": {
                "input_text": "test text",
                "embedding_length": len(test_embedding),
                "embedding_sample": test_embedding[:5] if test_embedding else []
            }
        }
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return {"status": "error", "detail": str(e)}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred, please try again later"},
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {APP_NAME} server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG) 