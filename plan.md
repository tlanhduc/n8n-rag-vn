I'll update the plan.md with our refined understanding of token limits and chunking strategy, without including the code implementations:

```markdown
# Vietnamese RAG Implementation Plan

## 1. Project Structure (unchanged)
```
vietnamese-rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── preprocessor.py  # Text preprocessing pipeline
│   │   └── embeddings.py    # Embedding generation
│   └── api/
│       ├── __init__.py
│       └── routes.py        # API endpoints
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md
```

## 2. Core Components:

### A. Data Models
- ProcessingRequest with configurable chunk parameters
  - chunk_size: default 110 tokens (safety margin below 128)
  - chunk_overlap: fixed at 20 tokens
- ProcessingResponse with chunks and embeddings
- QueryRequest with configurable top_k
- QueryResponse with matches and scores

### B. Text Preprocessing Service
Key Features:
- Text normalization using underthesea
- Sentence segmentation
- Word segmentation with domain-specific fixed words
- Chunking strategy:
  - Target chunk size: 110 tokens (safety margin below 128)
  - Fixed overlap: 20 tokens
  - Special handling for long sentences
  - Validation to ensure chunks never exceed 128 tokens

### C. Embedding Service using sentence-transformers
Key Features:
- Uses bkai-foundation-models/vietnamese-bi-encoder
- Input validation for 128 token limit
- Batch processing for efficiency
- Caching for frequent queries
- CPU optimization for VPS deployment

## 3. Implementation Requirements:
```
fastapi>=0.68.0
uvicorn>=0.15.0
underthesea>=1.3.0
sentence-transformers>=2.2.0
numpy>=1.21.0
pydantic>=1.8.0
python-dotenv>=0.19.0
```

## 4. Implementation Steps:

1. **Setup and Configuration**:
   - Virtual environment setup
   - Dependency installation
   - Configuration management
   - Logging setup

2. **Preprocessing Pipeline Implementation**:
   - Text normalization with underthesea
   - Word segmentation with fixed_words list
   - Implement chunking strategy:
     - Maintain 110 token chunk size
     - Ensure 20 token overlap
     - Validate against 128 token limit
   - Add error handling and validation

3. **Embedding Service Implementation**:
   - Setup sentence-transformers
   - Implement token limit validation
   - Add caching system
   - Optimize batch processing
   - Add error handling

4. **API Development**:
   - FastAPI endpoints implementation
   - Request/response validation
   - Error handling
   - API documentation
   - Rate limiting for VPS

5. **Testing and Optimization**:
   - Unit tests for components
   - Integration tests
   - Token limit validation tests
   - Chunking strategy tests
   - Performance testing
   - Memory optimization

## 5. VPS Deployment Considerations:

1. **Resource Management**:
   - Memory monitoring for embedding model
   - Batch size optimization
   - Request queuing system
   - Comprehensive logging

2. **Performance Optimization**:
   - Embedding cache implementation
   - Chunk size optimization
   - Resource cleanup
   - Health check system

3. **Scaling Strategy**:
   - Horizontal scaling plan
   - Load balancing setup
   - Monitoring system
   - Backup procedures

## 6. Token Management Strategy:

1. **Chunking Rules**:
   - Maximum chunk size: 110 tokens (safety margin)
   - Fixed overlap: 20 tokens
   - Never exceed 128 token limit

2. **Validation Layers**:
   - Preprocessor validation
   - Embedding service validation
   - API request validation

3. **Error Handling**:
   - Token limit exceeded errors
   - Invalid input handling
   - Chunk size violations

4. **Monitoring**:
   - Token usage tracking
   - Chunk size distribution
   - Embedding generation times
   - Cache hit rates

Would you like me to elaborate on any specific section of this updated plan?
