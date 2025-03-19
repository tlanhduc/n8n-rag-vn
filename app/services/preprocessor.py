import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from underthesea import word_tokenize, sent_tokenize, text_normalize
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


class TextPreprocessor:
    def __init__(self, fixed_words: Optional[List[str]] = None):
        """
        Initialize the text preprocessor.
        
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
        Normalize text using underthesea's text_normalize and clean whitespace.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Use underthesea's text normalization
        text = text_normalize(text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def segment_words(self, text: str) -> List[str]:
        """
        Segment Vietnamese text into words while preserving fixed words.
        
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
        
        # Segment words - by default, word_tokenize returns a list of tokens
        segmented_tokens = word_tokenize(text_with_placeholders)
        
        # Restore fixed words in the tokens
        if placeholders:
            # If word_tokenize returned a list, process each token
            if isinstance(segmented_tokens, list):
                for i, token in enumerate(segmented_tokens):
                    for placeholder, word in placeholders.items():
                        if placeholder in token:
                            segmented_tokens[i] = token.replace(placeholder, word)
            # If it returned a string (in case of format="text"), process the string
            elif isinstance(segmented_tokens, str):
                for placeholder, word in placeholders.items():
                    segmented_tokens = segmented_tokens.replace(placeholder, word)
                # Convert to list by splitting on spaces
                segmented_tokens = segmented_tokens.split()
        
        return segmented_tokens
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using underthesea's sent_tokenize.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        sentences = sent_tokenize(text)
        return sentences
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in Vietnamese text accurately by first segmenting words.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Accurate token count for Vietnamese text
        """
        if not text:
            return 0
        
        # Simply get the list of tokens and count them
        tokens = self.segment_words(text)
        return len(tokens)
    
    def process_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
        """
        Main method to process text into chunks with proper overlap.
        
        Args:
            text: Raw text to process
            chunk_size: Target size of each chunk in tokens (default: 110)
            chunk_overlap: Number of tokens to overlap between chunks (default: 20)
            
        Returns:
            List of chunk dictionaries, each with text and metadata
        """
        # Normalize text
        normalized_text = self.normalize_text(text)
        
        # Segment into sentences
        sentences = self.segment_sentences(normalized_text)
        
        # Create chunks with proper overlap
        chunks = self._create_chunks_from_sentences(sentences, chunk_size, chunk_overlap)
        
        # Validate and adjust chunk sizes if needed
        chunks = self._validate_chunk_sizes(chunks)
        
        return chunks

    def _create_chunks_from_sentences(self, sentences: List[str], 
                                     chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Create chunks from a list of sentences with proper overlap.
        
        Args:
            sentences: List of sentences to process
            chunk_size: Max token count for each chunk
            chunk_overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk_tokens = []
        current_size = 0
        
        for sentence in sentences:
            # Get tokens for this sentence
            sentence_tokens = self.segment_words(sentence)
            sentence_token_count = len(sentence_tokens)
            
            if sentence_token_count > chunk_size:
                # Handle long sentences separately
                self._handle_long_sentence(chunks, current_chunk_tokens, current_size, 
                                          sentence_tokens, chunk_size)
                # Reset current chunk tracking
                current_chunk_tokens = []
                current_size = 0
                continue
            
            if current_size + sentence_token_count > chunk_size:
                # Finish current chunk and start new one with overlap
                chunks.append(self._create_chunk(current_chunk_tokens, current_size))
                # Create overlap for next chunk
                current_chunk_tokens, current_size = self._create_overlap(
                    current_chunk_tokens, current_size, chunk_overlap)
            
            # Add sentence to current chunk
            current_chunk_tokens.extend(sentence_tokens)
            current_size += sentence_token_count
        
        # Add the final chunk if not empty
        if current_chunk_tokens:
            chunks.append(self._create_chunk(current_chunk_tokens, current_size))
        
        return chunks

    def _handle_long_sentence(self, chunks: List[Dict[str, Any]], 
                             current_chunk_tokens: List[str], current_size: int,
                             sentence_tokens: List[str], chunk_size: int) -> None:
        """
        Handle sentences that are longer than the chunk size.
        
        Args:
            chunks: List of chunks to append to
            current_chunk_tokens: Tokens in the current chunk
            current_size: Size of current chunk in tokens
            sentence_tokens: Tokens of the long sentence
            chunk_size: Maximum chunk size
        """
        # First save any existing chunk
        if current_chunk_tokens:
            chunks.append(self._create_chunk(current_chunk_tokens, current_size))
        
        # Split long sentence into parts
        current_part = []
        current_part_tokens = 0
        
        for token in sentence_tokens:
            if current_part_tokens + 1 <= chunk_size:
                current_part.append(token)
                current_part_tokens += 1
            else:
                # Save current part and start a new one
                chunks.append(self._create_chunk(current_part, current_part_tokens))
                current_part = [token]
                current_part_tokens = 1
        
        # Save any remaining part
        if current_part:
            chunks.append(self._create_chunk(current_part, current_part_tokens))

    def _create_overlap(self, tokens: List[str], size: int, 
                       overlap_size: int) -> Tuple[List[str], int]:
        """
        Create overlap for the next chunk.
        
        Args:
            tokens: Tokens from the previous chunk
            size: Size of the previous chunk
            overlap_size: Number of tokens to overlap
            
        Returns:
            Tuple of (overlap_tokens, overlap_size)
        """
        overlap_size = min(overlap_size, size)
        if overlap_size <= 0:
            return [], 0
        
        # Take tokens from the end for overlap
        tokens_to_keep = []
        tokens_kept = 0
        
        for token in reversed(tokens):
            if tokens_kept < overlap_size:
                tokens_to_keep.insert(0, token)
                tokens_kept += 1
            else:
                break
            
        return tokens_to_keep, tokens_kept

    def _create_chunk(self, tokens: List[str], token_count: int) -> Dict[str, Any]:
        """
        Create a chunk dictionary from tokens.
        
        Args:
            tokens: List of tokens to include in the chunk
            token_count: Number of tokens
            
        Returns:
            Chunk dictionary with text and metadata
        """
        return {
            "text": " ".join(tokens),
            "metadata": {
                "token_count": token_count
            }
        }

    def _validate_chunk_sizes(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate that all chunks are within the token limit.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of validated/adjusted chunks
        """
        validated_chunks = []
        
        for chunk in chunks:
            token_count = chunk["metadata"]["token_count"]
            
            if token_count > MAX_TOKEN_LIMIT:
                # Re-tokenize and truncate if necessary
                text = chunk["text"]
                tokens = self.segment_words(text)[:MAX_TOKEN_LIMIT]
                
                validated_chunks.append({
                    "text": " ".join(tokens),
                    "metadata": {
                        "token_count": len(tokens),
                        "truncated": True
                    }
                })
            else:
                validated_chunks.append(chunk)
            
        return validated_chunks 