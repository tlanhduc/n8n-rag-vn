#!/usr/bin/env python
import sys
import json
import logging
from services.preprocessor import TextPreprocessor
from services.embeddings import EmbeddingService
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MAX_TOKEN_LIMIT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Run a simple test of the Vietnamese RAG system."""
    logger.info("Testing Vietnamese RAG preprocessing and embedding...")
    
    # Sample Vietnamese text for testing - longer sample to demonstrate chunking
    sample_text = """
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
    Khi các đoạn văn bản đã được chia nhỏ, chúng sẽ được chuyển đổi thành các vector embedding 
    để có thể tìm kiếm ngữ nghĩa hiệu quả.
    
    Underthesea là một thư viện xử lý ngôn ngữ tự nhiên cho tiếng Việt được phát triển bởi nhóm 
    nghiên cứu Underthesea. Thư viện này cung cấp nhiều công cụ hữu ích như phân đoạn từ (word segmentation),
    phân đoạn câu (sentence segmentation), chuẩn hóa văn bản (text normalization), và nhiều chức năng khác.
    
    Khi phát triển hệ thống RAG cho tiếng Việt, chúng ta cần đảm bảo rằng các đoạn văn bản được chia nhỏ
    có kích thước không quá 128 token để phù hợp với giới hạn của mô hình embedding. Đồng thời, các đoạn 
    văn bản cần có sự chồng lấp (overlap) để đảm bảo tính liên tục của ngữ nghĩa.
    """
    
    # Initialize services
    logger.info("Initializing services...")
    preprocessor = TextPreprocessor()
    embedding_service = EmbeddingService(preprocessor=preprocessor)
    
    # Step 1: Normalize text
    logger.info("Step 1: Normalizing text...")
    normalized_text = preprocessor.normalize_text(sample_text)
    print(f"\nNormalized text (first 200 chars):\n{normalized_text[:200]}...\n")
    
    # Step 2: Count tokens
    logger.info("Step 2: Counting tokens...")
    token_count = preprocessor.count_tokens(normalized_text)
    print(f"Total token count: {token_count}\n")
    
    # Step 3: Process text into chunks
    logger.info(f"Step 3: Processing text into chunks (size={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP})...")
    chunks = preprocessor.process_text(
        text=normalized_text,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    
    print(f"Created {len(chunks)} chunks\n")
    
    # Print detailed chunk information
    print("=== CHUNK DETAILS ===")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({chunk['metadata']['token_count']} tokens):")
        print(f"Text: {chunk['text'][:100]}...\n")
        
        # Show visualization of chunk coverage
        if i < len(chunks) - 1:
            # Find overlap with next chunk
            current_words = set(chunk['text'].split())
            next_words = set(chunks[i+1]['text'].split())
            overlap_words = current_words.intersection(next_words)
            
            print(f"Overlap with next chunk: {len(overlap_words)} words")
            
            # Show some overlap words
            if overlap_words:
                print(f"Sample overlap words: {list(overlap_words)[:5]}")
        print("-" * 80)
    
    # Step 4: Generate embeddings
    logger.info("Step 4: Generating embeddings...")
    chunks_with_embeddings = embedding_service.embed_chunks(chunks)
    
    # Print embedding dimensions
    embedding_dim = len(chunks_with_embeddings[0]['embedding'])
    print(f"\nEmbedding dimensions: {embedding_dim}\n")
    
    # Step 5: Demonstrate the RAG query process
    logger.info("Step 5: Testing similarity search...")
    query = "Xử lý ngôn ngữ tự nhiên tiếng Việt"
    print(f"\nQuery: \"{query}\"\n")
    
    # Segment and count tokens in the query
    segmented_query = preprocessor.segment_words(query)
    query_tokens = len(segmented_query)  # Now just use len since segment_words returns a list
    print(f"Segmented query: {' '.join(segmented_query)}")
    print(f"Query token count: {query_tokens}\n")
    
    # Check token limit
    if query_tokens > MAX_TOKEN_LIMIT:
        print(f"WARNING: Query exceeds max token limit ({query_tokens} > {MAX_TOKEN_LIMIT})")
        print("Would need to be chunked for production use.")
    
    try:
        # Perform similarity search
        embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]
        texts = [chunk["text"] for chunk in chunks_with_embeddings]
        metadata = [chunk["metadata"] for chunk in chunks_with_embeddings]
        
        matches = embedding_service.similarity_search(
            query=query,
            embeddings=embeddings,
            texts=texts,
            metadata=metadata,
            top_k=2
        )
        
        # Display search results
        print("=== SEARCH RESULTS ===")
        for i, match in enumerate(matches):
            print(f"Match {i+1} (score: {match['score']:.4f}):")
            print(f"Text: {match['text'][:150]}...\n")
            print(f"Token count: {match['metadata']['token_count']}")
            print("-" * 80)
    except ValueError as e:
        print(f"ERROR: {str(e)}")
    
    logger.info("Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 