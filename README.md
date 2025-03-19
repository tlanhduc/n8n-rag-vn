# Vietnamese RAG Implementation

A Vietnamese language Retrieval Augmented Generation (RAG) system with specialized text processing and embeddings for Vietnamese language.

## Features

- Text normalization using underthesea
- Word segmentation with domain-specific fixed words
- Smart chunking strategy (110 tokens with 20 token overlap)
- Embedding generation using bkai-foundation-models/vietnamese-bi-encoder
- API for processing documents and querying similar chunks

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   uvicorn main:app --reload
   ```

## API Endpoints

- `POST /api/process`: Process text documents into chunks and embeddings
- `POST /api/query`: Find similar chunks for a given query text

## License

[MIT License](LICENSE) 