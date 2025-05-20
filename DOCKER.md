# Docker Implementation for Vietnamese RAG Application

This document provides instructions for building, running, and configuring the Vietnamese RAG application using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/n8n-rag-vn.git
   cd n8n-rag-vn
   ```

2. Create a `.env` file with your configuration (or copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

3. Customize the port in the `.env` file if needed:
   ```
   PORT=24600  # Change this to your desired port
   ```

4. Build and start the application using Docker Compose:
   ```bash
   docker compose up -d
   ```

5. Access the API at http://localhost:${PORT}/docs (where ${PORT} is the port specified in your .env file, default is 8000)

## Configuration

### Environment Variables

The Docker container can be configured using the `PORT` environment variable in the `.env` file.

### Volumes

The Docker container uses a volume to persist the downloaded models:

- `huggingface_cache`: Persistent storage for downloaded models

## Troubleshooting

If you encounter issues with the Docker implementation, try the following:

1. Check the container logs:
   ```bash
   docker logs vietnamese-rag-app
   ```

2. Ensure the PORT environment variable is correctly set in your `.env` file.

3. If the container fails to start, try rebuilding the image:
   ```bash
   docker compose build --no-cache
   docker compose up -d
   ```


