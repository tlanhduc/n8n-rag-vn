# Vietnamese RAG Implementation

A Vietnamese language Retrieval Augmented Generation (RAG) system with specialized text processing and embeddings for Vietnamese language.

## Features

- Text normalization using `underthesea`
- Sentence segmentation
- Word segmentation with domain-specific fixed words (optional)
- Smart chunking strategy with configurable chunk size and overlap (default: 110 tokens with 20 token overlap)
- Embedding generation using `bkai-foundation-models/vietnamese-bi-encoder`
- API for processing documents and querying similar chunks
- Caching for embeddings (optional, enabled by default)
- Input validation to ensure chunk size and overlap constraints

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   (Note: The `--host` and `--port` are optional and default to `0.0.0.0` and `8000` respectively, as defined in `app/config.py`)

## API Endpoints

- `POST /api/process`: Process text documents into chunks and embeddings.  Takes a `ProcessingRequest` as input, allowing specification of `chunk_size` and `chunk_overlap`. Returns a list of `EmbeddingResponse`.
- `POST /api/query`: Find similar chunks for a given query text. Takes a `QueryRequest` and returns a `QueryResponse`.
- `GET /api/status`:  Get server status.
- `GET /health`: Health check endpoint.
- `GET /`: Root endpoint with basic application information.

## Configuration

Configuration options are managed in `app/config.py` and can be overridden using environment variables:

- `DEBUG`: Enable debug mode (default: `False`)
- `EMBEDDING_MODEL`:  The SentenceTransformer model to use (default: `bkai-foundation-models/vietnamese-bi-encoder`)
- `MAX_TOKEN_LIMIT`: Maximum number of tokens per chunk (default: 128)
- `DEFAULT_CHUNK_SIZE`: Default chunk size in tokens (default: 110)
- `DEFAULT_CHUNK_OVERLAP`: Default chunk overlap in tokens (default: 20)
- `DEFAULT_TOP_K`: Default number of top matches to return for a query (default: 5)
- `ENABLE_CACHE`: Enable embedding caching (default: `True`)
- `CACHE_SIZE`: Maximum size of the embedding cache (default: 1000)
- `HOST`: Host address (default: `0.0.0.0`)
- `PORT`: Port number (default: 8000)

## License

[MIT License](LICENSE) 