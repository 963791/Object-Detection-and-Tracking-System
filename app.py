# app.py
import os
import tempfile
import base64
import time
import numpy as np
import cv2
import gradio as gr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------
# Config
# -----------------------
WEIGHTS_PATH = os.getenv("MODEL_PATH", "weights/yolov8n.pt")
CONF_DEFAULT = float(os.getenv("CONF", 0.45))
IMG_SIZE = int(os.getenv("IMG_SIZE", 640))
MAX_TRAIL_LEN = int(os.getenv("MAX_TRAIL_LEN", 30))

# -----------------------
# Utilities
# -----------------------
def id_to_color(track_id):
    np.random.seed(int(track_id) + 12345)
    c = tuple(np.random.randint(0, 256, 3).tolist())
    return (int(c[0]), int(c[1]), int(c[2]))

def center_from_ltrb(l, t, r, b):
    return int((l + r) / 2), int((t + b) / 2)

# -----------------------
# Model & Tracker load
# -----------------------
if os.path.exists(WEIGHTS_PATH):
    model = YOLO(WEIGHTS_PATH)
else:
    model = YOLO("yolov8n.pt")

tracker = DeepSort(max_iou_distance=0.7, max_age=30, n_init=1)
TRAILS = {}

# -----------------------
# Core detection + tracking per-frame (BGR in, BGR out)
# -----------------------
def detect_and_track_frame(frame_bgr, conf=CONF_DEFAULT, show_labels=True, show_ids=True, show_trail=True):
    if frame_bgr is None:
        return None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, conf=conf, imgsz=IMG_SIZE, verbose=False)[0]

    detections = []
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            detections.append(([int(x1), int(y1), int(x2), int(y2)], float(score), str(int(cls))))

    tracks = tracker.update_tracks(detections, frame=rgb)

    annotated = frame_bgr.copy()
    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = track.to_ltrb()
        l, t, r, b = map(int, (l, t, r, b))
        tid = track.track_id

        cx, cy = center_from_ltrb(l, t, r, b)
        TRAILS.setdefault(tid, []).append((cx, cy))
        if len(TRAILS[tid]) > MAX_TRAIL_LEN:
            TRAILS[tid] = TRAILS[tid][-MAX_TRAIL_LEN:]

        color = id_to_color(tid)
        cv2.rectangle(annotated, (l, t), (r, b), color, 2)

        # label text
        cls_name = ""
        try:
            cls_name = model.model.names[int(track.det_class)]
        except Exception:
            cls_name = str(track.det_class)
        label_parts = []
        if show_labels:
            label_parts.append(str(cls_name))
        if show_ids:
            label_parts.append(f"ID:{tid}")
        label = " ".join(label_parts).strip()

        if label:
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(annotated, (l, t - 20), (l + t_size[0] + 6, t), color, -1)
            cv2.putText(annotated, label, (l + 3, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # trail lines
        if show_trail and tid in TRAILS and len(TRAILS[tid]) > 1:
            pts = TRAILS[tid]
            for i in range(1, len(pts)):
                pt1 = pts[i-1]; pt2 = pts[i]
                alpha = i / len(pts)
                col = (int(color[0]*alpha), int(color[1]*alpha), int(color[2]*alpha))
                cv2.line(annotated, pt1, pt2, col, 2)

    return annotated

# -----------------------
# Gradio handlers
# -----------------------
def process_image(image, conf, show_labels, show_ids, show_trail):
    if image is None:
        return None
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated = detect_and_track_frame(bgr, conf=conf, show_labels=show_labels, show_ids=show_ids, show_trail=show_trail)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def process_webcam(frame, conf, show_labels, show_ids, show_trail):
    if frame is None:
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated = detect_and_track_frame(bgr, conf=conf, show_labels=show_labels, show_ids=show_ids, show_trail=show_trail)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def process_video(video_file, conf, show_labels, show_ids, show_trail):
    if video_file is None:
        return None
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp_out.name
    tmp_out.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    TRAILS.clear()
    tracker.max_age = 30; tracker.n_init = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = detect_and_track_frame(frame, conf=conf, show_labels=show_labels, show_ids=show_ids, show_trail=show_trail)
        writer.write(annotated)
    cap.release(); writer.release()
    return out_path

# -----------------------
# Build Gradio UI
# -----------------------
with gr.Blocks(title="YOLOv8 + DeepSORT (Realtime)") as demo:
    gr.Markdown("# 🎯 Object Detection & Tracking\n**Webcam · Image · Video** — YOLOv8n + DeepSORT")
    with gr.Row():
        with gr.Column(scale=3):
            tabs = gr.Tabs()
            with tabs:
                with gr.TabItem("Image Upload"):
                    img_in = gr.Image(label="Upload Image (RGB)")
                    img_conf = gr.Slider(label="Confidence", minimum=0.1, maximum=0.9, value=CONF_DEFAULT, step=0.05)
                    img_labels = gr.Checkbox(value=True, label="Show class labels")
                    img_ids = gr.Checkbox(value=True, label="Show tracking IDs")
                    img_trail = gr.Checkbox(value=True, label="Show trails")
                    img_out = gr.Image(label="Result")
                    btn = gr.Button("Detect & Track Image")
                    btn.click(fn=process_image, inputs=[img_in, img_conf, img_labels, img_ids, img_trail], outputs=[img_out])

                with gr.TabItem("Webcam (Live)"):
                    cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input")
                    cam_conf = gr.Slider(label="Confidence (webcam)", minimum=0.1, maximum=0.9, value=CONF_DEFAULT, step=0.05)
                    cam_labels = gr.Checkbox(value=True, label="Show class labels")
                    cam_ids = gr.Checkbox(value=True, label="Show tracking IDs")
                    cam_trail = gr.Checkbox(value=True, label="Show trails")
                    cam_out = gr.Image(label="Webcam Output")
                    cam.stream(fn=process_webcam, inputs=[cam, cam_conf, cam_labels, cam_ids, cam_trail], outputs=[cam_out])

                with gr.TabItem("Video Upload"):
                    vid_in = gr.Video(label="Upload a short video (mp4, avi)")
                    vid_conf = gr.Slider(label="Confidence (video)", minimum=0.1, maximum=0.9, value=CONF_DEFAULT, step=0.05)
                    vid_labels = gr.Checkbox(value=True, label="Show class labels")
                    vid_ids = gr.Checkbox(value=True, label="Show tracking IDs")
                    vid_trail = gr.Checkbox(value=True, label="Show trails")
                    vid_out = gr.Video(label="Processed Video")
                    btn_vid = gr.Button("Process Video")
                    btn_vid.click(fn=process_video, inputs=[vid_in, vid_conf, vid_labels, vid_ids, vid_trail], outputs=[vid_out])

        with gr.Column(scale=1):
            gr.Markdown("### Controls & Notes")
            gr.Markdown(
                """
- Webcam runs in-browser and sends frames to the server for processing.
- For faster FPS, reduce image size (IMG_SIZE) or increase server CPU.
- Processed videos are returned as downloadable MP4 files.
- You can include `weights/yolov8n.pt` in `weights/` or let Ultralytics auto-download on first run.
"""
            )

    gr.Markdown("## Model & Tracker")
    gr.Markdown(f"- YOLO weights path: `{WEIGHTS_PATH}`")
    gr.Markdown("- Tracker: deep-sort-realtime (IOU mode)")

if __name__ == "__main__":
    demo.queue(concurrency_count=2)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
