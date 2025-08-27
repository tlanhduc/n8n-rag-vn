import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from app.config import MAX_TOKEN_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# List of domain-specific fixed words that shouldn't be segmented
FIXED_WORDS = [
    "COVID-19",
    "AI",
    "NLP",
    "RAG",
    # Add more domain-specific terms here
]


class SimpleTextPreprocessor:
    def __init__(self, fixed_words: Optional[List[str]] = None):
        """
        Initialize the text preprocessor (simplified version without underthesea).
        
        Args:
            fixed_words: List of domain-specific fixed words that shouldn't be segmented
        """
        self.fixed_words = fixed_words or FIXED_WORDS
        # Compile a regex pattern for fixed words to avoid segmentation
        self.fixed_words_pattern = self._compile_fixed_words_pattern()
        
    def _compile_fixed_words_pattern(self) -> re.Pattern:
        """Compile a regex pattern for fixed words to avoid segmentation."""
        if not self.fixed_words:
            return re.compile(r"^$")  # Empty pattern
        
        # Escape special characters and join with OR
        escaped_words = [re.escape(word) for word in self.fixed_words]
        pattern = r"\b(" + "|".join(escaped_words) + r")\b"
        return re.compile(pattern, re.IGNORECASE)
    
    def normalize_text(self, text: str) -> str:
        """
        Simple text normalization without underthesea.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Simple text cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        
        return text
    
    def segment_words(self, text: str) -> List[str]:
        """
        Simple word segmentation without underthesea.
        
        Args:
            text: Text to segment
            
        Returns:
            List of segmented words
        """
        if not text:
            return []
        
        # Extract fixed words and replace with placeholders
        placeholders = {}
        def replace_with_placeholder(match):
            word = match.group(0)
            placeholder = f"__FIXED_WORD_{len(placeholders)}__"
            placeholders[placeholder] = word
            return placeholder
        
        text_with_placeholders = self.fixed_words_pattern.sub(replace_with_placeholder, text)
        
        # Simple word segmentation by space
        words = text_with_placeholders.split()
        
        # Restore fixed words
        for i, word in enumerate(words):
            for placeholder, original_word in placeholders.items():
                if placeholder in word:
                    words[i] = word.replace(placeholder, original_word)
        
        return words
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Simple sentence segmentation without underthesea.
        
        Args:
            text: Text to segment into sentences
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence segmentation by common sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_chunks(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Create text chunks with specified size and overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        
        # Validate chunk parameters
        if chunk_size > MAX_TOKEN_LIMIT:
            raise ValueError(f"Chunk size {chunk_size} exceeds maximum limit {MAX_TOKEN_LIMIT}")
        
        if chunk_overlap >= chunk_size:
            raise ValueError(f"Chunk overlap {chunk_overlap} must be less than chunk size {chunk_size}")
        
        # Segment text into sentences first
        sentences = self.segment_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.segment_words(sentence))
            
            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Keep some sentences for overlap
                    overlap_tokens = 0
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        sent_tokens = len(self.segment_words(sent))
                        if overlap_tokens + sent_tokens <= chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_tokens += sent_tokens
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
