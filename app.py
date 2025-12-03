# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Initialize Models
# -----------------------------
st.set_page_config(page_title="Object Detection & Tracking", layout="wide")
st.title("🟢 Object Detection & Tracking System")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

# -----------------------------
# Input Options
# -----------------------------
input_type = st.radio("Select Input Type:", ["Webcam", "Image Upload", "Video Upload"])

# -----------------------------
# Helper Functions
# -----------------------------
def detect_objects(frame):
    results = model(frame)[0]
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def track_objects(frame):
    results = model(frame)[0]  # YOLO predictions
    detections = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        score = 0.99  # or results.boxes.confidence if available
        detections.append(([x1, y1, x2, y2], score))  # ✅ Correct format

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame



# -----------------------------
# 1️⃣ Webcam Input
# -----------------------------
if input_type == "Webcam":
    webcam_frame = st.camera_input("Capture from Webcam")
    if webcam_frame is not None:
        # Convert to OpenCV format
        img = np.array(Image.open(webcam_frame))
        img = detect_objects(img)  # Object detection
        img = track_objects(img)   # Object tracking
        st.image(img, channels="RGB")

# -----------------------------
# 2️⃣ Image Upload
# -----------------------------
elif input_type == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file))
        img = detect_objects(img)
        img = track_objects(img)
        st.image(img, channels="RGB")

# -----------------------------
# 3️⃣ Video Upload
# -----------------------------
elif input_type == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_video is not None:
        tfile = uploaded_video.name
        with open(tfile, 'wb') as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect_objects(frame)
            frame = track_objects(frame)
            stframe.image(frame, channels="RGB")
        cap.release()
