import streamlit as st
import cv2
import time
import tempfile
from ultralytics import YOLO
# output run command:streamlit run app1.py


st.title("🚗 Car Detection (YOLOv8 + Streamlit)")

# Sidebar: choose video source
video_source = st.sidebar.radio("Choose video source:", ("Upload File", "Webcam"))

# Load YOLO model
model = YOLO("yolov8n.pt")

# Handle video input
if video_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
    else:
        st.warning("Please upload a video file to proceed.")
        st.stop()
else:
    video_path = 0  # webcam

cap = cv2.VideoCapture(video_path)

# Optional: Stop automatically after a fixed duration (e.g., 60 seconds)
max_duration = 60  # seconds
start_time = time.time()

# Streamlit video display
stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Filter only cars and draw bounding boxes
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == "car":
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Convert BGR (OpenCV) to RGB (Streamlit)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame in Streamlit
    stframe.image(frame_rgb, channels="RGB")

    # Stop after max_duration
    if time.time() - start_time > max_duration:
        st.info("Stopped automatically after 60 seconds.")
        break

cap.release()
