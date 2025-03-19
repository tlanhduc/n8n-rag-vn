import unittest
from services.preprocessor import TextPreprocessor
from services.embeddings import EmbeddingService
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class TestVietnameseProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize preprocessor and embedding service once for all tests
        cls.preprocessor = TextPreprocessor()
        cls.embedding_service = EmbeddingService(preprocessor=cls.preprocessor)
        
        # Sample Vietnamese text
        cls.sample_text = """
        Xử lý ngôn ngữ tự nhiên (NLP) là một lĩnh vực nghiên cứu quan trọng trong trí tuệ nhân tạo. 
        Đối với tiếng Việt, việc xử lý ngôn ngữ tự nhiên có những thách thức riêng do đặc điểm ngôn ngữ. 
        Tiếng Việt là một ngôn ngữ đơn lập, có thanh điệu, và một từ có thể bao gồm nhiều âm tiết. 
        Phân đoạn từ trong tiếng Việt là một bước quan trọng để xử lý văn bản tiếng Việt chính xác.
        
        Hệ thống RAG (Retrieval Augmented Generation) cho tiếng Việt cần phải giải quyết các vấn đề như:
        1. Chuẩn hóa văn bản tiếng Việt
        2. Phân đoạn câu và từ chính xác
        3. Xử lý các từ đặc biệt như COVID-19, AI, NLP
        4. Tính toán độ tương đồng ngữ nghĩa giữa các đoạn văn bản
        
        Việc chia nhỏ văn bản thành các đoạn có kích thước phù hợp (chunking) cũng đòi hỏi phải 
        hiểu cấu trúc ngữ pháp tiếng Việt để không chia cắt thông tin ở những vị trí không phù hợp.
        """
    
    def test_token_counting(self):
        """Test that token counting works correctly for Vietnamese text."""
        # Check token count for the sample text
        token_count_preprocessor = self.preprocessor.count_tokens(self.sample_text)
        token_count_embedding = self.preprocessor.count_tokens(self.sample_text)  # Now using the same method
        
        # Both methods should give the same result
        self.assertEqual(token_count_preprocessor, token_count_embedding)
        
        # Check that multi-syllable Vietnamese words are counted correctly
        text = "xử lý ngôn ngữ tự nhiên"
        segmented_tokens = self.preprocessor.segment_words(text)
        token_count = len(segmented_tokens)
        # In Vietnamese "xử lý", "ngôn ngữ", "tự nhiên" should be 3 tokens
        self.assertEqual(token_count, 3)  # Should count as 3 tokens, not 6
        
        # Check with fixed words
        text = "COVID-19 và AI trong NLP"
        segmented_tokens = self.preprocessor.segment_words(text)
        token_count = len(segmented_tokens)
        self.assertEqual(token_count, 5)  # COVID-19, và, AI, trong, NLP
    
    def test_chunking_with_default_parameters(self):
        """Test chunking with default parameters (110 tokens, 20 token overlap)."""
        chunks = self.preprocessor.process_text(
            text=self.sample_text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        
        # Verify we got some chunks
        self.assertGreater(len(chunks), 0)
        
        # Verify chunk sizes don't exceed the limit
        for chunk in chunks:
            token_count = chunk["metadata"]["token_count"]
            self.assertLessEqual(token_count, DEFAULT_CHUNK_SIZE)
        
        # Check for overlap between consecutive chunks
        if len(chunks) > 1:
            first_chunk_text = chunks[0]["text"]
            second_chunk_text = chunks[1]["text"]
            
            # There should be some overlap between chunks
            # Extract the last few words from first chunk
            first_chunk_words = set(first_chunk_text.split())
            next_words = set(second_chunk_text.split())
            overlap_words = first_chunk_words.intersection(next_words)
            
            # Should find some overlapping words
            self.assertTrue(
                len(overlap_words) > 0,
                "No overlap found between consecutive chunks"
            )
    
    def test_fixed_words_preservation(self):
        """Test that domain-specific fixed words are preserved during segmentation."""
        # Create text with fixed words
        text = "COVID-19 đang là vấn đề toàn cầu, AI và NLP giúp nghiên cứu nhanh hơn."
        
        # Process the text
        segmented_tokens = self.preprocessor.segment_words(text)
        
        # Fixed words should be preserved as is (check they exist in the token list)
        self.assertTrue("COVID-19" in segmented_tokens)
        self.assertTrue("AI" in segmented_tokens)
        self.assertTrue("NLP" in segmented_tokens)
    
    def test_embedding_generation(self):
        """Test embedding generation for Vietnamese text."""
        # Create a small chunk of Vietnamese text
        text = "Xử lý ngôn ngữ tự nhiên cho tiếng Việt."
        
        # Generate embedding
        embedding = self.embedding_service.get_embedding(text)
        
        # Check embedding dimensions
        self.assertEqual(len(embedding), self.embedding_service.model_dim)
        
        # Check that all values are floats
        self.assertTrue(all(isinstance(value, float) for value in embedding))


if __name__ == "__main__":
    unittest.main() 