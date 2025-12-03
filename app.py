# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

st.set_page_config(page_title="Object Detection (OID-capable) + Tracking", layout="wide")
st.title("Object Detection + Tracking — (Upload custom OID model for bottles/sunglasses)")

# ---------------- Sidebar: model & settings ----------------
st.sidebar.header("Model & Detection Settings")

# Model path: allow user to upload weights or use a default
uploaded_weights = st.sidebar.file_uploader("Upload custom YOLO .pt weights (optional)", type=["pt"])
use_default = st.sidebar.checkbox("Use default YOLO (COCO) if no custom weights", value=True)

conf_thres = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.5, 0.05)
iou_thres = st.sidebar.slider("NMS IoU threshold", 0.3, 0.6, 0.45, 0.05)
imgsz = st.sidebar.selectbox("Image size (px)", [640, 768, 896], index=0)  # larger => better accuracy, slower

st.sidebar.markdown("---")
st.sidebar.write("If you want improved bottle/sunglasses detection, upload an OID-pretrained or custom-trained `.pt` file here.")

# ---------------- Load model (from uploaded weights or default) ----------------
@st.cache_resource
def load_yolo(weights_path: str | None):
    if weights_path and os.path.exists(weights_path):
        model = YOLO(weights_path)
    else:
        # fallback to ultralytics default (COCO) model
        model = YOLO("yolov8m.pt")
    return model

# If user uploaded weights, save them to a temp file and pass path
weights_path = None
if uploaded_weights is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp.write(uploaded_weights.read())
    tmp.flush()
    weights_path = tmp.name
else:
    if not use_default:
        st.sidebar.warning("No weights provided and default disabled — upload a .pt file or enable default.")
        # we'll still load default to avoid crashing
        weights_path = None

model = load_yolo(weights_path)

# initialize tracker (DeepSORT)
@st.cache_resource
def init_tracker():
    return DeepSort(max_age=30, n_init=1, max_cosine_distance=0.2)

tracker = init_tracker()

# Helper: run detection and return results object
def run_yolo(frame_bgr):
    # Ultralytics expects RGB, but their API accepts numpy BGR too; we'll pass frame and set params
    results = model.predict(
        source=frame_bgr,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=imgsz,
        device="cpu",
        verbose=False
    )
    return results[0]  # Results for single image

# Helper: format model label to clean English
def get_label_name(class_id):
    # model.names is a dict mapping id -> label (already English for pretrained models)
    names = model.names if hasattr(model, "names") else {}
    return names.get(int(class_id), str(class_id))

# Simple temporal smoothing structure (for video): keep last label for each bbox center
recent_labels = {}

def detect_and_track_frame(frame):
    """Run YOLO detection + DeepSORT tracking on a single frame."""

    # -------------------------
    # 1. YOLOv8 DETECTION
    # -------------------------
    results = model.predict(frame, conf=0.45, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.model.names[cls]  # Clean English class names

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": label
            })

    # -------------------------
    # 2. UPDATE TRACKER
    # -------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    # -------------------------
    # 3. DRAW DETECTIONS + TRACKS
    # -------------------------
    for t in tracks:
        if not t.is_confirmed():
            continue

        det = None

        # SAFE ACCESS (prevents crash)
        if hasattr(t, "last_detection") and isinstance(t.last_detection, dict):
            det = t.last_detection

        if det is None:
            continue

        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        conf = det["confidence"]
        track_id = t.track_id  # Track ID

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label text
        text = f"ID {track_id} | {cls} {conf:.2f}"

        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    return frametated

# ---------------- UI: Input modes ----------------
st.markdown("**Choose input:**")
mode = st.selectbox("", ("One-shot Webcam", "Image Upload", "Video Upload"))

if mode == "One-shot Webcam":
    cam = st.camera_input("Capture an image (click to allow webcam)")
    if cam:
        img_pil = Image.open(cam).convert("RGB")
        frame = np.array(img_pil)[:, :, ::-1].copy()  # RGB->BGR
        out = detect_and_track_frame(frame)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

elif mode == "Image Upload":
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        frame = np.array(img_pil)[:, :, ::-1].copy()
        out = detect_and_track_frame(frame)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

elif mode == "Video Upload":
    uploaded = st.file_uploader("Upload video (mp4/mov/avi)", type=["mp4","mov","avi"])
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
