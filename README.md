
# 🎯 Object Detection and Tracking System

This is a real-time **object detection and tracking** web app using **YOLOv8** and **Deep SORT**, deployed with **Streamlit**. It detects multiple objects in uploaded video files and assigns tracking IDs using Deep SORT.

---

## 🚀 Features

- ✅ Object detection using **pre-trained YOLOv8**
- ✅ Object tracking using **Deep SORT**
- ✅ Bounding boxes with **class labels** and **unique tracking IDs**
- ✅ Streamlit-based interactive UI
- ✅ Upload and process your own videos

---

## 📁 Project Structure

CodeAlpha_object-Detection-and-Tracking-/
├── app.py # Streamlit UI code
├── yolo_tracker.py # YOLO + Deep SORT logic
├── requirements.txt # Python dependencies
├── runtime.txt # Python version
├── README.md # Project documentation
└── models/
└── yolov8n.pt # YOLO model weights (optional local copy)

---

## 🔧 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/963791/CodeAlpha_object-Detection-and-Tracking-.git
cd CodeAlpha_object-Detection-and-Tracking-
pip install -r requirements.txt
streamlit run app.py
