# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import time
import os
import socket

# Detection & tracking
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Optional live webcam
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBSOCKET_AVAILABLE = True
except Exception:
    WEBSOCKET_AVAILABLE = False

# -------------------------
# App config & UI
# -------------------------
st.set_page_config(page_title="Object Detection + Tracking", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f6f9ff 0%, #fff6fb 100%); }
.header { background: linear-gradient(90deg,#7dd3fc,#fbcfe8); padding: 12px; border-radius: 12px; text-align:center;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h2>🎯 Object Detection + Tracking</h2></div>', unsafe_allow_html=True)
st.write("Professional UI, YOLOv8 + DeepSORT. Sidebar for modes & settings.")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("YOLO model", ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt"))
    conf_thres = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    max_cosine_distance = st.slider("Tracker max cosine distance", 0.1, 1.0, 0.2, 0.05)
    max_iou_distance = st.slider("Tracker max IOU distance", 0.3, 0.9, 0.7, 0.05)
    max_age = st.number_input("Tracker max_age (frames)", min_value=1, max_value=300, value=30, step=1)
    display_fps = st.checkbox("Show FPS (live webcam)", value=True)
    mode = st.radio("Mode", ["Image Upload", "Video Upload", "One-Shot Webcam", "Live Webcam"])
    st.write("---")
    st.markdown("Notes: Live webcam works **only locally**. Online deployment (Streamlit Cloud) supports only image/video upload & one-shot webcam.")

# -------------------------
# Load model & tracker
# -------------------------
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

@st.cache_resource
def init_tracker(max_cosine_distance, max_iou_distance, max_age):
    return DeepSort(max_age=max_age, n_init=1, max_cosine_distance=max_cosine_distance, max_iou_distance=max_iou_distance)

model = load_model(model_choice)
tracker = init_tracker(max_cosine_distance, max_iou_distance, max_age)

# -------------------------
# Utility functions
# -------------------------
def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2 = [int(x) for x in t.to_ltrb()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 127, 80), 2)
        cv2.putText(frame, f"ID {t.track_id}", (x1, max(20, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return frame

def detect_and_track(frame, model, tracker, conf_thres):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, conf=conf_thres, imgsz=640, verbose=False)
    r = results[0]
    boxes = []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (bb, cf, cl) in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = bb.tolist()
            boxes.append([int(x1), int(y1), int(x2), int(y2), float(cf), int(cl)])
    ds_dets = []
    for b in boxes:
        x1, y1, x2, y2, conf, cls = b
        ds_dets.append(([x1, y1, x2, y2], conf, {"class": int(cls)}))
    tracks = tracker.update_tracks(ds_dets, frame=frame)
    frame = draw_tracks(frame, tracks)
    return frame

# -------------------------
# Environment check
# -------------------------
def is_local():
    """Detect if running on local machine."""
    hostname = socket.gethostname()
    if hostname.lower() in ["localhost", "127.0.0.1"]:
        return True
    # Streamlit Cloud sets "streamlit" in environment
    return os.environ.get("STREAMLIT_SERVER") is None

LOCAL = is_local()

# -------------------------
# Mode handlers
# -------------------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        out = detect_and_track(img, model, tracker, conf_thres)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

elif mode == "Video Upload":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4","mov","avi","mkv"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        play = st.button("Process Video")
        video_placeholder = st.empty()
        if play:
            frames = []
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            writer = cv2.VideoWriter(out_temp.name, fourcc, fps, (width, height))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            frame_idx = 0
            pbar = st.progress(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                proc = detect_and_track(frame, model, tracker, conf_thres)
                writer.write(proc)
                frame_idx += 1
                if frame_idx % 5 == 0:
                    pbar.progress(min(frame_idx/total, 1.0))
            writer.release()
            cap.release()
            st.success("Processing Done!")
            st.video(out_temp.name)

elif mode == "One-Shot Webcam":
    cam = st.camera_input("Capture Image")
    if cam:
        img_pil = Image.open(cam).convert("RGB")
        img = np.array(img_pil)[:, :, ::-1].copy()
        out = detect_and_track(img, model, tracker, conf_thres)
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)

elif mode == "Live Webcam":
    if not LOCAL or not WEBSOCKET_AVAILABLE:
        st.warning("Live webcam streaming works only locally. Use 'One-Shot Webcam' or uploads online.")
        st.stop()

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.tracker = init_tracker(max_cosine_distance, max_iou_distance, max_age)
            self.model = load_model(model_choice)
            self.prev_time = time.time()
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            out_frame = detect_and_track(img, self.model, self.tracker, conf_thres)
            if display_fps:
                now = time.time()
                fps = 1.0 / (now - self.prev_time + 1e-8)
                self.prev_time = now
                cv2.putText(out_frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            return out_frame

    webrtc_streamer(key="object-detection", mode="recvonly", rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=VideoProcessor)

st.markdown("---")
st.caption("Powered by YOLOv8 + DeepSORT. Live webcam works only locally; online supports one-shot webcam & uploads.")
