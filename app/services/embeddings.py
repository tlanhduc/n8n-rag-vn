import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL, 
    MAX_TOKEN_LIMIT, 
    ENABLE_CACHE, 
    CACHE_SIZE
)
from app.services.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, preprocessor=None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            preprocessor: Optional TextPreprocessor instance
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.model_dim}")
        
        # Use provided preprocessor or create one
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Set up caching if enabled
        if ENABLE_CACHE:
            self.get_embedding = lru_cache(maxsize=CACHE_SIZE)(self._get_embedding)
        else:
            self.get_embedding = self._get_embedding
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for embedding generation")
            return [0.0] * self.model_dim
        
        # Use preprocessor for token counting only
        token_count = self.preprocessor.count_tokens(text)
        
        # Check against token limit
        if token_count > MAX_TOKEN_LIMIT:
            logger.error(
                f"Text exceeds max token limit ({token_count} > {MAX_TOKEN_LIMIT}). "
                f"Please chunk your text before encoding."
            )
            raise ValueError(f"Text exceeds max token limit ({token_count} > {MAX_TOKEN_LIMIT})")
            
        try:
            # Directly encode the text string
            embedding = self.model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.model_dim
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Validate texts are within token limit
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                logger.warning(f"Empty or invalid text at index {i}")
                continue
                
            # Check token count
            token_count = self.preprocessor.count_tokens(text)
            if token_count > MAX_TOKEN_LIMIT:
                logger.error(
                    f"Text at index {i} exceeds max token limit ({token_count} > {MAX_TOKEN_LIMIT}). "
                    f"Please chunk your text before encoding."
                )
                raise ValueError(f"Text at index {i} exceeds max token limit ({token_count} > {MAX_TOKEN_LIMIT})")
        
        try:
            # Let the model handle the batch encoding directly
            embeddings = self.model.encode(texts).tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [[0.0] * self.model_dim] * len(texts)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            
        Returns:
            List of chunk dictionaries with added embeddings
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.get_embeddings_batch(texts)
        
        # Add embeddings to chunks
        result_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding
            result_chunks.append(chunk_with_embedding)
        
        return result_chunks
    
    def similarity_search(
        self, 
        query: str, 
        embeddings: List[List[float]], 
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar texts to a query.
        
        Args:
            query: Query text
            embeddings: List of embedding vectors to search
            texts: List of texts corresponding to the embeddings
            metadata: Optional list of metadata for each text
            top_k: Number of top matches to return
            
        Returns:
            List of matches with text, score, and metadata
        """
        if not query or not embeddings or not texts:
            return []
        
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]
        
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        
        # Convert to numpy arrays for efficient computation
        query_embedding_np = np.array(query_embedding)
        embeddings_np = np.array(embeddings)
        
        # Compute cosine similarity
        similarity_scores = np.dot(embeddings_np, query_embedding_np) / (
            np.linalg.norm(embeddings_np, axis=1) * np.linalg.norm(query_embedding_np)
        )
        
        # Get top-k indices
        if top_k > len(texts):
            top_k = len(texts)
            
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                "text": texts[idx],
                "score": float(similarity_scores[idx]),
                "metadata": metadata[idx]
            })
        
        return results 