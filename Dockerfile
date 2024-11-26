# Sử dụng hình ảnh Python làm cơ sở
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép toàn bộ mã nguồn vào container
COPY . /app

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Mở port cho ứng dụng Streamlit
EXPOSE 8501

# Lệnh để chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
