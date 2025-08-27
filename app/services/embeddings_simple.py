import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, ENABLE_CACHE, CACHE_SIZE

logger = logging.getLogger(__name__)


class SimpleEmbeddingService:
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding service with a simplified approach.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self.model = None
        self.cache = {} if ENABLE_CACHE else None
        self.cache_size = CACHE_SIZE if ENABLE_CACHE else 0
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a simple hash-based embedding for testing
            self.model = None
            logger.warning("Using fallback hash-based embedding method")
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """
        Fallback embedding method using simple hash-based approach.
        This is only for testing when the main model fails to load.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        # Simple hash-based embedding for testing
        import hashlib
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a list of floats
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            float_val = float(int(hex_pair, 16)) / 255.0  # Normalize to 0-1
            embedding.append(float_val)
        
        # Pad or truncate to get a consistent embedding size
        target_size = 384  # Common embedding size
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return embedding
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        if not text:
            return []
        
        # Check cache first
        if self.cache is not None and text in self.cache:
            return self.cache[text]
        
        try:
            if self.model is not None:
                # Use the actual model
                embedding = self.model.encode(text).tolist()
            else:
                # Use fallback method
                embedding = self._get_fallback_embedding(text)
            
            # Cache the result
            if self.cache is not None:
                if len(self.cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return fallback embedding
            return self._get_fallback_embedding(text)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of tuples (index, similarity_score)
        """
        if not query_embedding or not candidate_embeddings:
            return []
        
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return similarities[:top_k]
