import sys
import os
import unittest
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embeddings import EmbeddingService
from services.preprocessor import TextPreprocessor
from config import EMBEDDING_MODEL, MAX_TOKEN_LIMIT

class TestEmbeddingService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = TextPreprocessor()
        cls.embedding_service = EmbeddingService(preprocessor=cls.preprocessor)
        
        # Process sample text through preprocessor
        sample_text = (
            "Transformer là kiến trúc mạng neural tiên tiến. "
            "BERT và GPT sử dụng kiến trúc này để xử lý ngôn ngữ tự nhiên. "
            "Các mô hình AI hiện đại đạt được kết quả ấn tượng trong NLP."
        )
        cls.chunks = cls.preprocessor.process_text(sample_text)

    def test_embedding_generation(self):
        # Test single embedding
        embedding = self.embedding_service.get_embedding("ví dụ về embedding")
        print("test_embedding_generation", embedding)
        print("test_embedding_generation length", len(embedding))
        print("test_embedding_generation model_dim", self.embedding_service.model_dim)
        self.assertEqual(len(embedding), self.embedding_service.model_dim)
        
        # Test batch embeddings
        embeddings = self.embedding_service.get_embeddings_batch(["text 1", "text 2"])
        self.assertEqual(len(embeddings), 2)
        
    def test_chunk_embedding(self):
        embedded_chunks = self.embedding_service.embed_chunks(self.chunks)
        print("test_chunk_embedding", embedded_chunks)
        self.assertEqual(len(embedded_chunks), len(self.chunks))
        
        for chunk in embedded_chunks:
            self.assertIn("embedding", chunk)
            self.assertEqual(len(chunk["embedding"]), self.embedding_service.model_dim)

    def test_token_limit_enforcement(self):
        # Create long text that exceeds token limit
        long_text = " ".join(["token"] * (MAX_TOKEN_LIMIT + 10))
        
        with self.assertRaises(ValueError):
            self.embedding_service.get_embedding(long_text)
            
        with self.assertRaises(ValueError):
            self.embedding_service.get_embeddings_batch([long_text])

    def test_similarity_search(self):
        # Embed sample chunks
        embedded_chunks = self.embedding_service.embed_chunks(self.chunks)
        
        # Perform similarity search
        query = "kiến trúc transformer"
        results = self.embedding_service.similarity_search(
            query=query,
            embeddings=[c["embedding"] for c in embedded_chunks],
            texts=[c["text"] for c in embedded_chunks],
            metadata=[c["metadata"] for c in embedded_chunks]
        )
        print("test_similarity_search", results)
        self.assertGreater(len(results), 0)
        self.assertIn("transformer", results[0]["text"].lower())

if __name__ == "__main__":
    unittest.main() 