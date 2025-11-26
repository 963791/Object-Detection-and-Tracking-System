import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import os
from deep_sort_realtime.deepsort_tracker import DeepSort


# =======================
# LOAD MODEL
# =======================

MODEL_PATH = "weights/yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8n weights...")
    model = YOLO("yolov8n.pt")
else:
    model = YOLO(MODEL_PATH)

tracker = DeepSort(max_age=30)


# =======================
# DETECTION + TRACKING
# =======================

def detect_and_track(input_frame):
    if input_frame is None:
        return None

    frame = input_frame.copy()
    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf)
        cls = int(box.cls)
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return frame


# =======================
# GRADIO UI
# =======================

def process_video(video):
    cap = cv2.VideoCapture(video)
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked = detect_and_track(frame)
        output_frames.append(tracked)

    cap.release()
    return output_frames


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎥 Advanced Object Detection + Tracking  
        ### YOLOv8 + DeepSORT | Webcam | Image | Video
        """
    )

    with gr.Tab("Webcam"):
        webcam = gr.Image(type="numpy", label="Webcam Stream", streaming=True)
        webcam_out = gr.Image(type="numpy", label="Output")
        webcam.stream(detect_and_track, inputs=webcam, outputs=webcam_out)

    with gr.Tab("Image Upload"):
        img_input = gr.Image(type="numpy", label="Upload Image")
        img_output = gr.Image(type="numpy")
        img_btn = gr.Button("Detect")
        img_btn.click(detect_and_track, inputs=img_input, outputs=img_output)

    with gr.Tab("Video Upload"):
        video_in = gr.Video(label="Upload Video")
        video_out = gr.Video(label="Processed Video")
        vid_btn = gr.Button("Process Video")
        vid_btn.click(process_video, inputs=video_in, outputs=video_out)

demo.launch(server_name="0.0.0.0", server_port=7860)