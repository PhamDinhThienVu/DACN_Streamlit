from ultralytics import YOLO

try:
    model = YOLO('./models/new-best.pt')  # Đường dẫn đầy đủ tới mô hình của bạn
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")


