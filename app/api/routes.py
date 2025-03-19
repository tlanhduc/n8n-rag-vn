from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
from functools import lru_cache

from app.models.schemas import (
    ProcessingRequest, 
    ProcessingResponse, 
    QueryRequest, 
    QueryResponse,
    TextChunk,
    ChunkData,
    QueryMatch,
    NormalizationResponse,
    BaseResponse,
    EmbeddingResponse
)
from app.services.preprocessor import TextPreprocessor
from app.services.embeddings import EmbeddingService
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, MAX_TOKEN_LIMIT

router = APIRouter()
logger = logging.getLogger(__name__)

# Updated dependency injection with caching
@lru_cache
def get_preprocessor():
    return TextPreprocessor()

@lru_cache
def get_embedding_service(preprocessor: TextPreprocessor = Depends(get_preprocessor)):
    return EmbeddingService(preprocessor=preprocessor)

# In-memory storage for chunks and embeddings
# In a production system, this would be replaced by a vector database
stored_chunks = []


@router.get("/status")
async def get_status():
    return BaseResponse(
        status="ok",
        message="Server is running"
    )

@router.post("/process", response_model=List[EmbeddingResponse])
async def process_text(
    request: ProcessingRequest,
    preprocessor: TextPreprocessor = Depends(get_preprocessor),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> List[EmbeddingResponse]:
    """
    Process text into chunks and generate embeddings.
    
    This endpoint:
    1. Normalizes the Vietnamese text using underthesea's text_normalize
    2. Segments the text into sentences using underthesea's sent_tokenize
    3. Segments words with underthesea's word_tokenize and fixed-word preservation
    4. Chunks the text with target size of 110 tokens and 20 token overlap 
       (or as specified in the request)
    5. Generates embeddings for each chunk using vietnamese-bi-encoder
    
    Each chunk is guaranteed to:
    - Not exceed 128 tokens (MAX_TOKEN_LIMIT)
    - Have proper overlap with adjacent chunks
    - Preserve sentence boundaries when possible
    """
    try:
        logger.info(f"Processing text of length {len(request.text)} with chunk_size={request.chunk_size}, chunk_overlap={request.chunk_overlap}")
        
        # Validate chunk parameters
        chunk_size = request.chunk_size or DEFAULT_CHUNK_SIZE
        chunk_overlap = request.chunk_overlap or DEFAULT_CHUNK_OVERLAP
        
        if chunk_size > MAX_TOKEN_LIMIT:
            logger.warning(f"Requested chunk_size {chunk_size} exceeds MAX_TOKEN_LIMIT {MAX_TOKEN_LIMIT}")
            chunk_size = MAX_TOKEN_LIMIT
            
        if chunk_overlap >= chunk_size:
            logger.warning(f"Requested chunk_overlap {chunk_overlap} is too large")
            chunk_overlap = chunk_size - 1
        
        # Process text into chunks
        chunks = preprocessor.process_text(
            text=request.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Text processed into {len(chunks)} chunks")
        
        # Log token counts for debugging
        token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
        logger.debug(f"Token counts per chunk: {token_counts}")
        logger.debug(f"Min tokens: {min(token_counts) if token_counts else 0}, Max tokens: {max(token_counts) if token_counts else 0}")
        
        # Calculate average chunk size
        avg_chunk_size = sum(token_counts) / len(token_counts) if token_counts else 0
        logger.info(f"Average chunk size: {avg_chunk_size:.1f} tokens")
        
        # Verify no chunks exceed the token limit
        for i, chunk in enumerate(chunks):
            token_count = chunk["metadata"]["token_count"]
            if token_count > MAX_TOKEN_LIMIT:
                logger.error(f"Chunk {i} exceeds token limit: {token_count} > {MAX_TOKEN_LIMIT}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Chunk {i} exceeds max token limit ({token_count} > {MAX_TOKEN_LIMIT})"
                )
        
        # Generate embeddings for chunks
        try:
            chunks_with_embeddings = embedding_service.embed_chunks(chunks)
        except ValueError as e:
            # Catch token limit errors from embedding service
            logger.error(f"Token limit error during embedding: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Add file metadata to each chunk
        for chunk in chunks_with_embeddings:
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            
            if request.file_id:
                chunk["metadata"]["file_id"] = request.file_id
            
            if request.file_title:
                chunk["metadata"]["file_title"] = request.file_title
        
        # Store chunks and embeddings in memory
        global stored_chunks
        stored_chunks = chunks_with_embeddings
        
        # Convert to response model
        chunk_objects = []
        for chunk in chunks_with_embeddings:
            chunk_objects.append(EmbeddingResponse(
                embedding=chunk["embedding"]
            ))
        
        return chunk_objects
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error processing text: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_similar(
    request: QueryRequest,
    preprocessor: TextPreprocessor = Depends(get_preprocessor),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> QueryResponse:
    """
    Find chunks similar to the query text.
    """
    try:
        global stored_chunks
        
        if not stored_chunks:
            raise HTTPException(status_code=400, detail="No chunks available. Process text first.")
        
        logger.info(f"Querying with '{request.query_text}', top_k={request.top_k}")
        
        # Verify query doesn't exceed token limit
        tokens = preprocessor.segment_words(request.query_text)
        token_count = len(tokens)
        if token_count > MAX_TOKEN_LIMIT:
            logger.error(f"Query exceeds token limit: {token_count} > {MAX_TOKEN_LIMIT}")
            raise HTTPException(
                status_code=400, 
                detail=f"Query exceeds max token limit ({token_count} > {MAX_TOKEN_LIMIT})"
            )
        
        # Extract embeddings and texts from stored chunks
        embeddings = [chunk["embedding"] for chunk in stored_chunks]
        texts = [chunk["text"] for chunk in stored_chunks]
        metadata = [chunk["metadata"] for chunk in stored_chunks]
        
        # Perform similarity search
        top_k = request.top_k or DEFAULT_TOP_K
        try:
            matches = embedding_service.similarity_search(
                query=request.query_text,
                embeddings=embeddings,
                texts=texts,
                metadata=metadata,
                top_k=top_k
            )
        except ValueError as e:
            # Catch token limit errors from embedding service
            logger.error(f"Token limit error during similarity search: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Convert to response model
        match_objects = []
        for match in matches:
            match_objects.append(QueryMatch(
                text=match["text"],
                score=match["score"],
                metadata=match["metadata"]
            ))
        
        logger.info(f"Found {len(match_objects)} matching chunks")
        
        return QueryResponse(
            matches=match_objects,
            total_matches=len(match_objects)
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error querying similar chunks: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error querying similar chunks: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error querying similar chunks: {str(e)}")


# New endpoint for text normalization
@router.post("/normalize", response_model=NormalizationResponse)
async def normalize_text(
    request: ProcessingRequest,  # Reuse the existing ProcessingRequest schema
    preprocessor: TextPreprocessor = Depends(get_preprocessor),
) -> NormalizationResponse:
    """
    Normalize, sentence segment, and word segment the input text.

    This endpoint:
    1. Normalizes Vietnamese text.
    2. Segments the text into sentences.
    3. Segments the sentences into words.
    """
    try:
        logger.info(f"Normalizing text of length {len(request.text)}")

        # 1. Normalize text
        normalized_text = preprocessor.normalize_text(request.text)

        # 2. Segment sentences
        sentences = preprocessor.segment_sentences(normalized_text)

        # 3. Segment words (preserving sentence structure)
        segmented_sentences = []
        for sentence in sentences:
            segmented_sentences.append(preprocessor.segment_words(sentence))

        return NormalizationResponse(
            # normalized_text=normalized_text,
            # sentences=sentences,
            segmented_sentences=segmented_sentences
        )

    except Exception as e:
        logger.error(f"Error normalizing text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error normalizing text: {str(e)}") 