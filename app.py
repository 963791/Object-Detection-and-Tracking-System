# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(page_title="Object Detection + Tracking (600+ Classes)", layout="wide")
st.title("Object Detection + Tracking — 600+ Classes (Objects365 Model)")

# ---------------- Sidebar: model & settings ----------------
st.sidebar.header("Model & Detection Settings")

uploaded_weights = st.sidebar.file_uploader("Upload YOLO .pt weights (optional)", type=["pt"])
use_default = st.sidebar.checkbox("Use 600+ Classes Default (Objects365)", value=True)

conf_thres = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.5, 0.05)
iou_thres = st.sidebar.slider("NMS IoU threshold", 0.3, 0.7, 0.45, 0.05)
imgsz = st.sidebar.selectbox("Image size (px)", [640, 768, 896, 1024], index=0)

st.sidebar.markdown("---")
st.sidebar.write("Default model: **YOLOv8x-Objects365 (600+ classes)** — detects spoons, sunglasses, bottles, etc.")


# ---------------- Load YOLO model ----------------
@st.cache_resource
def load_yolo(weights_path: str | None):
    if weights_path and os.path.exists(weights_path):
        return YOLO(weights_path)
    else:
        return YOLO("yolov8x-o365.pt")   # ← 600+ classes model


weights_path = None
if uploaded_weights:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp.write(uploaded_weights.read())
    tmp.flush()
    weights_path = tmp.name
else:
    if not use_default:
        st.sidebar.warning("No weights uploaded — default model enabled.")
        weights_path = None

model = load_yolo(weights_path)


# ---------------- Initialize DeepSORT tracker ----------------
@st.cache_resource
def init_tracker():
    return DeepSort(max_age=30, n_init=1, max_cosine_distance=0.2)

tracker = init_tracker()


# ---------------- Detection + Tracking Function ----------------
def clean_label(name):
    return name.replace("_", " ").strip().title()

def detect_and_track_frame(frame):

    results = model.predict(
        frame,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=imgsz,
        verbose=False,
        device="cpu"
    )

    detections = []  # ([x1, y1, x2, y2], conf, label)

    r = results[0]
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        raw_name = model.names.get(cls_id, f"class_{cls_id}")
        cls_name = clean_label(raw_name)

        detections.append(([x1, y1, x2, y2], conf, cls_name))

    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results
    for t in tracks:
        if not t.is_confirmed():
            continue

        det = t.last_detection
        if det is None:
            continue

        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class"]
        conf = det["confidence"]
        track_id = t.track_id

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)

        # Label
        label = f"{cls_name} | ID:{track_id} | {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (50, 255, 50), 2)

    return frame


# ---------------- UI Input Modes ----------------
st.markdown("### Choose Input Source")

mode = st.selectbox("", ("Webcam Snapshot", "Image Upload", "Video Upload"))


# ---------- Webcam ----------
if mode == "Webcam Snapshot":
    cam = st.camera_input("Take a picture")
    if cam:
        img_pil = Image.open(cam).convert("RGB")
        frame = np.array(img_pil)[:, :, ::-1].copy()
        out = detect_and_track_frame(frame)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)


# ---------- Image Upload ----------
elif mode == "Image Upload":
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        frame = np.array(img_pil)[:, :, ::-1].copy()
        out = detect_and_track_frame(frame)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)


# ---------- Video Upload ----------
elif mode == "Video Upload":
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out = detect_and_track_frame(frame)
            stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        cap.release()
