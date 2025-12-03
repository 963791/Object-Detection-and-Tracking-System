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

def detect_and_track_frame(frame_bgr):
    """
    Input: OpenCV BGR frame
    Returns: annotated frame (BGR)
    """
    res = run_yolo(frame_bgr)
    detections_ds = []  # list of ([x1,y1,x2,y2], score, {optional metadata})

    # Collect detections
    if hasattr(res, "boxes") and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for (box, conf, cls) in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = map(int, box.tolist())
            # Put in DeepSORT format: ([x1,y1,x2,y2], score, metadata)
            detections_ds.append(([x1, y1, x2, y2], float(conf), {"class": int(cls)}))

    # Update tracker
    tracks = tracker.update_tracks(detections_ds, frame=frame_bgr)

    # Draw boxes and labels
    annotated = frame_bgr.copy()
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        track_id = t.track_id
        # Determine label: prefer the class from t.last_detection if exists (DeepSORT metadata pass-through)
        meta = t.det_confidence if hasattr(t, "det_confidence") else None
        # Try to extract class from t.last_detection (if available in the tracker implementation)
        cls = None
        if t.last_detection and isinstance(t.last_detection, dict):
            cls = t.last_detection.get("class", None)

        # As a fallback, try to match current detections that intersect this track bbox
        label = ""
        best_conf = 0.0
        for det in detections_ds:
            (dx1, dy1, dx2, dy2), score, det_meta = det
            # overlap check
            inter_x1 = max(x1, dx1); inter_y1 = max(y1, dy1)
            inter_x2 = min(x2, dx2); inter_y2 = min(y2, dy2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                if score > best_conf:
                    best_conf = score
                    cls = det_meta.get("class", cls)

        if cls is not None:
            label_name = get_label_name(cls)
        else:
            label_name = "object"

        # Temporal smoothing: avoid label flicker if confidence low
        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
        key = f"{track_id}"
        recent = recent_labels.get(key, (None, 0))  # (label, count)
        prev_label, prev_count = recent
        if prev_label is None:
            recent_labels[key] = (label_name, 1)
            chosen_label = label_name
        else:
            if label_name == prev_label:
                recent_labels[key] = (label_name, min(prev_count + 1, 10))
                chosen_label = label_name
            else:
                # Only switch if new label has higher confidence or repeated
                if best_conf >= 0.75:
                    recent_labels[key] = (label_name, 1)
                    chosen_label = label_name
                else:
                    # keep previous until new label becomes stable
                    recent_labels[key] = (prev_label, max(prev_count - 1, 0))
                    chosen_label = prev_label

        # Draw box + text
        color = (10, 200, 180)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        txt = f"{chosen_label} | ID:{track_id}"
        cv2.putText(annotated, txt, (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return annotated

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
