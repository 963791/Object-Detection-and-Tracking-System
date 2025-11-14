import cv2
import torch
import gradio as gr
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------------
# Load YOLOv8 Model
# ---------------------------
model = YOLO("yolov8n.pt")   # Use yolov8s.pt or custom model if needed

# ---------------------------
# Initialize DeepSORT Tracker
# ---------------------------
tracker = DeepSort(max_age=30,
                   n_init=2,
                   max_cosine_distance=0.3,
                   nn_budget=None,
                   embedder="mobilenet",
                   half=True)

# ---------------------------
# Detection + Tracking Function
# ---------------------------
def detect_and_track(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO detection
    results = model(frame, conf=0.5)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, cls))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        l, t, w, h = int(l), int(t), int(w), int(h)

        cv2.rectangle(frame, (l, t), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# ---------------------------
# Gradio Interface
# ---------------------------
def process_image(image):
    return detect_and_track(image)

def process_video(video):
    cap = cv2.VideoCapture(video)
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = detect_and_track(frame)
        output_frames.append(out)

    cap.release()
    return output_frames


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(source="upload", label="Upload Image / Use Camera", type="numpy"),
    outputs=gr.Image(type="numpy"),
    examples=[],
    live=False,
    title="YOLOv8 + DeepSORT Object Tracking",
    description="Real-time object detection & multi-object tracking"
)

# Add webcam
demo_cam = gr.Interface(
    fn=process_image,
    inputs=gr.Image(source="webcam", label="Webcam Input", type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
)

# Add video upload
demo_video = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Output Video"),
)

# Tabs UI
final_ui = gr.TabbedInterface(
    [demo, demo_cam, demo_video],
    ["Image Upload", "Live Webcam", "Video Upload"]
)

final_ui.launch()
