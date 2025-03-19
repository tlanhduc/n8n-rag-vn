from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, MAX_TOKEN_LIMIT
import uuid


class ProcessingRequest(BaseModel):
    """Request model for processing text into chunks and embeddings."""
    text: str = Field(..., description="The text to be processed")
    chunk_size: Optional[int] = Field(DEFAULT_CHUNK_SIZE, description="Target size of each chunk in tokens")
    chunk_overlap: Optional[int] = Field(DEFAULT_CHUNK_OVERLAP, description="Number of tokens to overlap between chunks")
    file_id: Optional[str] = Field(None, description="ID of the file being processed")
    file_title: Optional[str] = Field(None, description="Title of the file being processed")
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v > MAX_TOKEN_LIMIT:
            raise ValueError(f"chunk_size cannot exceed {MAX_TOKEN_LIMIT} tokens")
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if v < 0:
            raise ValueError("chunk_overlap cannot be negative")
        return v


class TextChunk(BaseModel):
    """Model representing a chunk of text with its embedding."""
    text: str
    embedding: List[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkData(BaseModel):
    """Model representing a processed chunk with its metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingResponse(BaseModel):
    """Response model for processed text chunks and embeddings."""
    chunks: List[ChunkData]
    total_chunks: int

class EmbeddingResponse(BaseModel):
    """Response model for embedding text."""
    embedding: List[float]
    

class QueryRequest(BaseModel):
    """Request model for querying similar chunks."""
    query_text: str = Field(..., description="The query text to find similar chunks for")
    top_k: Optional[int] = Field(DEFAULT_TOP_K, description="Number of top matches to return")
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v


class QueryMatch(BaseModel):
    """Model representing a matched chunk with similarity score."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response model for query results."""
    matches: List[QueryMatch]
    total_matches: int


class NormalizationResponse(BaseModel):
    # normalized_text: str
    # sentences: List[str]
    segmented_sentences: List[List[str]] 

class BaseResponse(BaseModel):
    status: str
    message: str
