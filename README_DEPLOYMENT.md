# Hướng dẫn Deploy lên Render.com

## Bước 1: Chuẩn bị Repository

Đảm bảo repository của bạn có các file sau:
- `render.yaml` - Cấu hình deployment
- `requirements.txt` - Python dependencies
- `runtime.txt` - Phiên bản Python
- `Procfile` - Lệnh khởi động
- `build.sh` - Script build

## Bước 2: Deploy lên Render

### 2.1. Đăng nhập Render
1. Truy cập [render.com](https://render.com)
2. Đăng nhập hoặc tạo tài khoản mới
3. Kết nối với GitHub repository

### 2.2. Tạo Web Service
1. Click "New +" → "Web Service"
2. Kết nối với repository `n8n-rag-vn`
3. Chọn branch `main`
4. Đặt tên service: `vietnamese-rag-api`

### 2.3. Cấu hình Service
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main_simple:app --host 0.0.0.0 --port $PORT`
- **Plan**: Free (hoặc Pro nếu cần)

### 2.4. Environment Variables
Render sẽ tự động sử dụng các biến từ `render.yaml`:
- `PORT`: 10000
- `HOST`: 0.0.0.0
- `DEBUG`: false
- `EMBEDDING_MODEL`: bkai-foundation-models/vietnamese-bi-encoder

## Bước 3: Deploy

1. Click "Create Web Service"
2. Render sẽ tự động build và deploy
3. Chờ quá trình hoàn tất (có thể mất 5-10 phút)
4. Service sẽ có URL dạng: `https://vietnamese-rag-api.onrender.com`

## Bước 4: Kiểm tra

### 4.1. Health Check
```bash
curl https://vietnamese-rag-api.onrender.com/health
```

### 4.2. API Status
```bash
curl https://vietnamese-rag-api.onrender.com/api/status
```

### 4.3. Test Endpoint
```bash
curl https://vietnamese-rag-api.onrender.com/api/test
```

## Bước 5: Sử dụng API

### 5.1. Process Text
```bash
curl -X POST "https://vietnamese-rag-api.onrender.com/api/process" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Đây là văn bản tiếng Việt để test",
    "chunk_size": 50,
    "chunk_overlap": 10
  }'
```

### 5.2. Query Similar
```bash
curl -X POST "https://vietnamese-rag-api.onrender.com/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "văn bản tiếng Việt",
    "top_k": 3
  }'
```

## Lưu ý quan trọng

1. **Free Plan Limitations**:
   - Service sẽ sleep sau 15 phút không hoạt động
   - Lần request đầu tiên sẽ mất thời gian để wake up
   - Giới hạn 750 giờ sử dụng/tháng

2. **Model Loading**:
   - Lần đầu tiên sẽ download model (~540MB)
   - Có thể mất 5-10 phút để hoàn tất

3. **Performance**:
   - Free plan có giới hạn CPU/RAM
   - Nên upgrade lên Pro plan cho production

## Troubleshooting

### Lỗi Build
- Kiểm tra `requirements.txt` có đúng dependencies
- Đảm bảo Python version trong `runtime.txt` tương thích

### Lỗi Runtime
- Kiểm tra logs trong Render dashboard
- Đảm bảo start command đúng
- Kiểm tra environment variables

### Lỗi Model Loading
- Kiểm tra internet connection
- Đảm bảo HuggingFace token nếu cần
- Kiểm tra disk space (free plan có giới hạn)

## Support

Nếu gặp vấn đề, hãy:
1. Kiểm tra logs trong Render dashboard
2. Kiểm tra GitHub Actions (nếu có)
3. Tạo issue trên GitHub repository
4. Liên hệ Render support
