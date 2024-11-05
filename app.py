
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from functions.detected_image import *;
import torch
import os
def load_model(model_path="./models/yolov9e.pt"):
    # Kiểm tra xem máy tính có GPU không
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model và chuyển sang GPU nếu có
    model = YOLO(model_path).to(device)
    return model





def main():
  st.title("Detect and Counting Object In An Image")

  st.sidebar.title("Settings")


  st.sidebar.markdown("---")

   # Cung cấp tùy chọn cho người dùng để chọn mô hình
  model_choice = st.sidebar.selectbox("Choose Model", ["YOLOv9e", "YOLOv9m", "YOLOv9Custom", "YOLOv9PretrainCustom"])

    # Cập nhật đường dẫn mô hình tương ứng
  if model_choice == "YOLOv9e":
    model_path = "./models/yolov9e.pt"  # Thay bằng đường dẫn tới model 
  elif model_choice == "YOLOv9m":
    model_path = "./models/yolov9m.pt"  # Thay bằng đường dẫn tới model 
  elif model_choice == "YOLOv9Custom":
    model_path = "./models/yolov9e.pt"  # Thay bằng đường dẫn tới model 
  else:
    model_path = "./models/yolov9e.pt"  # Đường dẫn tới mô hình tùy chỉnh nếu có

  # Tải mô hình
  # Kiểm tra nếu mô hình chưa được tải trong session_state, nếu chưa thì tải lại mô hình
  if 'model' not in st.session_state or st.session_state.model_path != model_path:
      st.session_state.model = load_model(model_path)
      st.session_state.model_path = model_path  # Lưu đường dẫn mô hình vào session state

  model = st.session_state.model  # Lấy mô hình đã tải từ session state

  # Thông tin mô hình
  st.sidebar.subheader("Model Information")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  st.sidebar.text(f"Using device: {device}")
  st.sidebar.text(f"Model: {model_choice}")
  st.sidebar.text(f"Model Path: {model_path}")
  st.sidebar.text(f"Number of Classes: {len(model.names)}")
  st.sidebar.text(f"Model Details: {model.info()}")

  st.sidebar.markdown("---")

  confident = st.sidebar.slider('Confidence Score: ', min_value = 0.0, max_value=1.0, value = 0.25)
  st.sidebar.markdown("---")
  
  ##Checkbox 
  # save_img = st.sidebar.checkbox("Save Images")
  # eable_GPU = st.sidebar.checkbox("Enable GPU")
  custom_classes  = st.sidebar.checkbox("Use Custom Classes")
  
  
  ##Custom class for use who can choose
  assigned_class_id = []
  coco_class_names = [
      'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
      'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 
      'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
      'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
      'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
      'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
      'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
      'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
      'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
      'toothbrush'
  ]
  if custom_classes:
    assigned_class = st.sidebar.multiselect("Select The Custom Classes", list(coco_class_names), default = coco_class_names)
    for each in assigned_class:
      assigned_class_id.append(coco_class_names.index(each))


    


  ## Hình ảnh đầu vào
  image_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


 
  if image_file_buffer is not None:
      try:
           # Đảm bảo rằng image_file_buffer là một đối tượng hợp lệ
            image = Image.open(image_file_buffer)
            st.sidebar.text("Input Image")
            st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
            # Resize ảnh nếu cần
            image_resized = image.resize((640, 640))

            if st.sidebar.button("Process"):   
            # Run YOLO model on the uploaded image
                with st.spinner("Processing... Please wait."):
                    class_counts, annotated_img = detect_and_count_objects(model, image, confident)
      
                    if class_counts is None:
                      st.warning("No objects detected! Maybe cause of threshold, check it!!!")
                    else:
            # Show processed image with bounding boxes
                      st.image(annotated_img, caption="Processed Image with Detected Objects", use_column_width=True)
            # Show object counts
                      st.write("### Object Counts by Class")
                      object_count_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
                      st.dataframe(object_count_df)     
            
      
      except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
  else:
      st.warning("Please upload an image to continue.")




  # Kiểm tra có ảnh đầu vào (từ upload hoặc ảnh mặc định)
  if st.session_state.input_image is not None:
      # Nếu có ảnh đầu vào thì hiển thị nút "Process"
      # --- Bổ sung nút "Process" ---
    if st.sidebar.button("Process"):   
            # Run YOLO model on the uploaded image
        with st.spinner("Processing... Please wait."):
                # Nhận diện và đếm số lượng object mỗi class
          image = Image.open(image_file_buffer)
          image_resized = image.resize((640, 640))
          class_counts, annotated_img = detect_and_count_objects(model, image, confident)
      
        if class_counts is None:
          st.warning("No objects detected! Maybe cause of threshold, check it!!!")
        else:
            # Show processed image with bounding boxes
          st.image(annotated_img, caption="Processed Image with Detected Objects", use_column_width=True)
            # Show object counts
          st.write("### Object Counts by Class")
          object_count_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
          st.dataframe(object_count_df)     

  else:
      st.sidebar.text("No image to process!")

if __name__ == '__main__':
  try:
    main()
  except SystemExit:
    pass




