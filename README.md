---
title: "Object Detection & Tracking System"
emoji: "🎯"
colorFrom: "indigo"
colorTo: "teal"
sdk: "gradio"
sdk_version: "4.44.1"
app_file: "app.py"
pinned: false
---

# Object Detection & Tracking (YOLOv8n + DeepSORT)

Features:
- Live webcam (in-browser) — real-time frames streamed to the server
- Upload image (annotated output)
- Upload video (processed MP4 returned)
- YOLOv8n for realtime detection (auto-downloads if weights missing)
- DeepSORT tracking (IDs + colored trails)
- Lightweight, tuned for deployment on Hugging Face Spaces

---

## How to run locally
1. Create venv and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
