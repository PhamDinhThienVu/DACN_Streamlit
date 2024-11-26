import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from io import BytesIO



def generate_class_colors(num_classes):
    colors = []
    for i in range(num_classes):
        color = np.random.randint(0, 256, size=3).tolist()
        colors.append(tuple(color))
    return colors

def detect_and_count_objects(model, image, confidence_threshold=0.25, classList=None):
    """
    Nhận đầu vào là ảnh và threshold, trả về ảnh đã bounding box và dataframe đếm số lượng object mỗi class.
    """
    # Dự đoán bounding boxes
    results = model.predict(source=image, conf=confidence_threshold, classes=classList)
    results_df = results[0].to_df()


     # Nếu không phát hiện được đối tượng nào
    if results_df.empty:
        return None, None  # Không có đối tượng nào, trả về None

    # Đếm số lượng đối tượng của mỗi lớp
    class_counts = results_df['name'].value_counts()

    
    # Vẽ bounding boxes lên ảnh
    img = np.array(image)  # Chuyển từ PIL Image sang numpy array
    class_colors = generate_class_colors(len(results_df['name'].unique()))

    # Vẽ các bounding boxes lên ảnh
    for _, row in results_df.iterrows():
        box = row['box']
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        class_name = row['name']
        confidence = row['confidence']
        
        # Lấy màu sắc cho lớp
        color = class_colors[results_df['name'].unique().tolist().index(class_name)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Vẽ với màu sắc lớp
        cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return class_counts, img