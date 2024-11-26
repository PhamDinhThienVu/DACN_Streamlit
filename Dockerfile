# Sử dụng Python image
FROM python:3.9-slim

# Cài đặt các thư viện cần thiết cho OpenCV và đồ họa
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglvnd0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép các tệp cần thiết vào container
COPY . /app

# Cài đặt các phụ thuộc Python
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng cho Flask hoặc Streamlit
EXPOSE 8501

# Chạy ứng dụng
CMD ["streamlit", "run", "app.py"]
