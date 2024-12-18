
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from functions.detected_image import *;
import torch
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

model_files = {
    "yolov9_custom.pt": "1rN0Z3y9ramnA21eNCj5aDo3QCJSMolLQ",  # Thay YOUR_FILE_ID_1 bằng ID Google Drive
    "yolov9_pretrain.pt": "1Zq5oAHIIRL9Ov1gCRod-bGmdOdn4bclC",
    "yolov9e.pt": "1gERgjMdPejtlqyfGzChH9DfpcjmzUH6z",
    "yolov9m.pt": "1oqGJVcvdjbT52p2tPgOiiQCwp02jrYj0",
}

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Tải mô hình nếu chưa tồn tại
for file_name, file_id in model_files.items():
    file_path = os.path.join(models_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        gdd.download_file_from_google_drive(file_id=file_id, dest_path=file_path, unzip=False)
        print(f"{file_name} downloaded successfully.")
    else:
        print(f"{file_name} already exists. Skipping download.")



def load_model(model_path="./models/yolov9e.pt"):
    # Kiểm tra xem máy tính có GPU không
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model và chuyển sang GPU nếu có
    model = YOLO(model_path).to(device)
    return model

def main():
  st.title("Detect and Counting Object In An Image")
  st.text("(For mobile) Click to the arow in the left coner to open the sidebar, choose ur options, close the sidebar and see the result ")
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
    model_path = "./models/yolov9_custom.pt"  # Thay bằng đường dẫn tới model 
  else:
    model_path = "./models/yolov9_pretrain.pt"  # Đường dẫn tới mô hình tùy chỉnh nếu có

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
  # Lấy danh sách lớp từ mô hình
  class_names = list(model.names.values())  # Lấy tên các lớp từ mô hình
  if custom_classes:
    assigned_class = st.sidebar.multiselect("Select The Custom Classes", list(class_names), default = class_names)
    for each in assigned_class:
      assigned_class_id.append(class_names.index(each))


    


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
                    class_counts, annotated_img = detect_and_count_objects(model, image, confident, assigned_class_id)
      
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

if __name__ == '__main__':
  try:
    main()
  except SystemExit:
    pass




