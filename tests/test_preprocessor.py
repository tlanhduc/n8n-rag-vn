import sys
import os
import unittest

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessor import TextPreprocessor
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MAX_TOKEN_LIMIT


class TestTextPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = TextPreprocessor()
        cls.sample_text = (
            "Trí tuệ nhân tạo (AI) đang cách mạng hóa ngành công nghệ. "
            "COVID-19 đã thúc đẩy quá trình chuyển đổi số. "
            "Các mô hình NLP như BERT và GPT-3 đạt được nhiều tiến bộ đáng kể."
        )
    
    def test_normalization(self):
        normalized = self.preprocessor.normalize_text("  Đây   là   ví   dụ  ")
        print("test_normalization", normalized)
        self.assertEqual(normalized, "Đây là ví dụ")
    
    def test_word_segmentation(self):
        tokens = self.preprocessor.segment_words(self.sample_text)
        print("test_word_segmentation", tokens)
        self.assertIn("AI", tokens)
        self.assertIn("COVID-19", tokens)
        self.assertIn("NLP", tokens)
    
    def test_token_counting(self):
        count = self.preprocessor.count_tokens(self.sample_text)
        print("test_token_counting", count)
        self.assertGreater(count, 15)
        self.assertLess(count, 50)
    
    def test_chunk_creation(self):
        chunks = self.preprocessor.process_text(self.sample_text)
        print("test_chunk_creation", chunks)
        self.assertGreater(len(chunks), 0)
        
        # Test chunk sizes
        for chunk in chunks:
            self.assertLessEqual(
                chunk["metadata"]["token_count"], 
                MAX_TOKEN_LIMIT,
                "Chunk exceeds maximum token limit"
            )
        
    def test_overlap_handling(self):
        chunks = self.preprocessor.process_text(
            self.sample_text,
            chunk_size=15,
            chunk_overlap=3
        )
        print("test_overlap_handling", chunks)
        # Verify overlap between consecutive chunks
        for i in range(1, len(chunks)):
            prev_words = set(self.preprocessor.segment_words(chunks[i-1]["text"]))
            current_words = set(self.preprocessor.segment_words(chunks[i]["text"]))
            overlap = prev_words & current_words
            self.assertGreaterEqual(len(overlap), 2)


if __name__ == "__main__":
    unittest.main() 