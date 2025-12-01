import os
import tempfile
import time
import traceback
import numpy as np
import cv2
import gradio as gr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------
# Config
# -------------------------
WEIGHTS_PATH = "weights/yolov8n.pt"  # optional: include this file in repo (use Git LFS) or let Ultralytics download
CONF_DEFAULT = 0.45
IMG_SIZE = 640
MAX_TRAIL_LEN = 30  # length of trail per track

# -------------------------
# Utilities
# -------------------------
def id_to_color(track_id: int):
    np.random.seed(int(track_id) + 12345)
    c = tuple(np.random.randint(0, 256, 3).tolist())
    return (int(c[0]), int(c[1]), int(c[2]))

def center_from_ltrb(l, t, r, b):
    return int((l + r) / 2), int((t + b) / 2)

def encode_jpg(frame_bgr):
    ok, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode jpg")
    return buf.tobytes()

# -------------------------
# Load model + tracker
# -------------------------
try:
    if os.path.exists(WEIGHTS_PATH):
        model = YOLO(WEIGHTS_PATH)
    else:
        # will auto-download to cache on first run
        model = YOLO("yolov8n.pt")
except Exception:
    # provide readable error in UI if model load fails
    model = None

# Lightweight DeepSORT (IOU mode by default)
tracker = DeepSort(max_iou_distance=0.7, max_age=30, n_init=1)
TRAILS = {}

# -------------------------
# Core: detect & track (BGR in, BGR out)
# -------------------------
def detect_and_track_frame(frame_bgr, conf=CONF_DEFAULT, show_labels=True, show_ids=True, show_trail=True):
    if model is None:
        raise RuntimeError("Model not loaded. Check logs.")
    if frame_bgr is None:
        return None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, conf=conf, imgsz=IMG_SIZE, verbose=False)[0]

    # collect detections for tracker: [bbox], score, class_name
    detections = []
    if hasattr(results, "boxes") and results.boxes is not None:
        # results.boxes.data: x1,y1,x2,y2,conf,cls
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            # deep_sort_realtime accepts [x1,y1,x2,y2], score and class label string
            try:
                cls_name = model.model.names[int(cls)]
            except Exception:
                cls_name = str(int(cls))
            detections.append(([int(x1), int(y1), int(x2), int(y2)], float(score), cls_name))

    # update tracker
    tracks = tracker.update_tracks(detections, frame=rgb)  # RGB frame also acceptable

    annotated = frame_bgr.copy()

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = track.to_ltrb()
        l, t, r, b = int(l), int(t), int(r), int(b)
        tid = track.track_id

        # update centroid trail
        cx, cy = center_from_ltrb(l, t, r, b)
        if tid not in TRAILS:
            TRAILS[tid] = []
        TRAILS[tid].append((cx, cy))
        if len(TRAILS[tid]) > MAX_TRAIL_LEN:
            TRAILS[tid] = TRAILS[tid][-MAX_TRAIL_LEN:]

        color = id_to_color(tid)

        # bounding box
        cv2.rectangle(annotated, (l, t), (r, b), color, 2)

        # label & id
        label_parts = []
        if show_labels and track.det_class is not None:
            try:
                cls_name = model.model.names[int(track.det_class)]
            except Exception:
                cls_name = str(track.det_class)
            label_parts.append(str(cls_name))
        if show_ids:
            label_parts.append(f"ID:{tid}")
        label = " ".join(label_parts).strip()

        if label:
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(annotated, (l, t - 20), (l + t_size[0] + 6, t), color, -1)
            cv2.putText(annotated, label, (l + 3, t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # draw fade trail
        if show_trail and tid in TRAILS and len(TRAILS[tid]) > 1:
            pts = TRAILS[tid]
            for i in range(1, len(pts)):
                pt1 = pts[i - 1]
                pt2 = pts[i]
                alpha = i / len(pts)
                col = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                cv2.line(annotated, pt1, pt2, col, 2)

    return annotated

# -------------------------
# Gradio handlers
# -------------------------
def process_image(image, conf, show_labels, show_ids, show_trail):
    # Gradio gives RGB numpy
    if image is None:
        return None
    try:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated = detect_and_track_frame(bgr, conf=conf, show_labels=show_labels, show_ids=show_ids, show_trail=show_trail)
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return None

def process_webcam(frame, conf, show_labels, show_ids, show_trail):
    # streaming webcam frames are RGB numpy
    if frame is None:
        return None
    try:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        annotated = detect_and_track_frame(bgr, conf=conf, show_labels=show_labels, show_ids=show_ids, show_trail=show_trail)
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def process_video(video_file, conf, show_labels, show_ids, show_trail):
    # returns a path to processed mp4
    if video_file is None:
        return None
    try:
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

        # reset trails/tracker for new video
        TRAILS.clear()
        tracker.max_age = 30
        tracker.n_init = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = detect_and_track_frame(frame, conf=conf, show_labels=show_labels, show_ids=show_ids, show_trail=show_trail)
            writer.write(annotated)

        cap.release()
        writer.release()
        return out_path
    except Exception as e:
        traceback.print_exc()
        return None

# -------------------------
# Build Gradio UI
# -------------------------
description = """
# 🎯 Object Detection & Tracking (YOLOv8n + DeepSORT)
- Use **Webcam**, **Upload Image** or **Upload Video**
- Toggle labels, tracking IDs and motion trails
"""

with gr.Blocks(title="Object Detection & Tracking", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=3):
            tabs = gr.Tabs()

            with tabs:
                with gr.TabItem("Image Upload"):
                    img_in = gr.Image(label="Upload image (RGB)", type="numpy")
                    img_conf = gr.Slider(label="Confidence", minimum=0.1, maximum=0.9, value=CONF_DEFAULT, step=0.05)
                    img_labels = gr.Checkbox(value=True, label="Show class labels")
                    img_ids = gr.Checkbox(value=True, label="Show tracking IDs")
                    img_trail = gr.Checkbox(value=True, label="Show trails")
                    img_out = gr.Image(label="Result", type="numpy")
                    btn_img = gr.Button("Detect & Track")
                    btn_img.click(fn=process_image, inputs=[img_in, img_conf, img_labels, img_ids, img_trail], outputs=[img_out])

                with gr.TabItem("Webcam (Live)"):
                    cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam (allow camera access)")
                    cam_conf = gr.Slider(label="Confidence (webcam)", minimum=0.1, maximum=0.9, value=CONF_DEFAULT, step=0.05)
                    cam_labels = gr.Checkbox(value=True, label="Show class labels")
                    cam_ids = gr.Checkbox(value=True, label="Show tracking IDs")
                    cam_trail = gr.Checkbox(value=True, label="Show trails")
                    cam_out = gr.Image(label="Webcam Output", type="numpy")
                    # streaming
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
- Webcam will ask browser permission; allow camera.
- For faster FPS reduce IMG_SIZE or use fewer options.
- If model downloads on first run it may take ~30s.
"""
            )

    gr.Markdown("## Model & Tracker")
    gr.Markdown(f"- YOLO weights path: `{WEIGHTS_PATH}`")
    gr.Markdown("- Tracker: deep-sort-realtime")

# Launch
if __name__ == "__main__":
    # queue allows handling concurrent requests
    demo.queue(concurrency_count=2)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
