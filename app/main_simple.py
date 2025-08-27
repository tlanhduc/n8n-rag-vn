from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
from app.config import APP_NAME, DEBUG, API_PREFIX, HOST, PORT
from app.services.preprocessor_simple import SimpleTextPreprocessor
from app.services.embeddings_simple import SimpleEmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO if not DEBUG else logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    debug=DEBUG,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize services
preprocessor = SimpleTextPreprocessor()
embedding_service = SimpleEmbeddingService()

# Pydantic models
class ProcessingRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class EmbeddingResponse(BaseModel):
    chunk: str
    embedding: List[float]
    chunk_index: int

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class QueryResponse(BaseModel):
    query: str
    results: List[dict]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": APP_NAME}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic application information."""
    return {
        "service": APP_NAME,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

# Status endpoint
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

# Process text endpoint
@app.post(f"{API_PREFIX}/process", response_model=List[EmbeddingResponse])
async def process_text(request: ProcessingRequest):
    """
    Process text documents into chunks and embeddings.
    
    Args:
        request: ProcessingRequest containing text and optional chunk parameters
        
    Returns:
        List of EmbeddingResponse with chunks and their embeddings
    """
    try:
        logger.info(f"Processing text with length: {len(request.text)}")
        
        # Create chunks
        chunks = preprocessor.create_chunks(
            request.text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings for chunks
        embeddings = embedding_service.get_embeddings_batch(chunks)
        
        # Create response
        responses = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            response = EmbeddingResponse(
                chunk=chunk,
                embedding=embedding,
                chunk_index=i
            )
            responses.append(response)
        
        logger.info(f"Successfully processed {len(responses)} chunks")
        return responses
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post(f"{API_PREFIX}/query", response_model=QueryResponse)
async def query_similar(request: QueryRequest):
    """
    Find similar chunks for a given query text.
    
    Args:
        request: QueryRequest containing query text and optional top_k parameter
        
    Returns:
        QueryResponse with query results
    """
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Generate embedding for query
        query_embedding = embedding_service.get_embedding(request.query)
        
        # For now, return a simple response since we don't have stored embeddings
        # In a real application, you would search against a database of embeddings
        
        results = [{
            "chunk": f"Sample response for query: {request.query}",
            "similarity": 0.95,
            "chunk_index": 0
        }]
        
        response = QueryResponse(
            query=request.query,
            results=results
        )
        
        logger.info("Query processed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {APP_NAME} on {HOST}:{PORT}")
    uvicorn.run(
        "app.main_simple:app",
        host=HOST,
        port=PORT,
        reload=DEBUG
    )
