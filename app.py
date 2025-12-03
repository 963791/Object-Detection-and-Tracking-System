import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tempfile

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

st.title("Object Detection & Tracking")

# Select input type
input_type = st.radio("Select Input Type:", ("Webcam", "Image Upload", "Video Upload"))

# ---------------- Webcam ----------------
if input_type == "Webcam":
    img_file_buffer = st.camera_input("Capture an image from your webcam")
    
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        results = model.predict(image)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(int(class_ids[i])), (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        st.image(image, channels="BGR")

# ---------------- Image Upload ----------------
elif input_type == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        results = model.predict(image)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(int(class_ids[i])), (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        st.image(image, channels="BGR")

# ---------------- Video Upload ----------------
elif input_type == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(int(class_ids[i])), (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            
            stframe.image(frame, channels="BGR")
        
        cap.release()
