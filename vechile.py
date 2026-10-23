import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
# output run command :streamlit run app2.py

# Title
st.title(" Vehicle Detection using  (YOLOv8 + Streamlit)")

# Sidebar options
st.sidebar.header("Video Input Options")
video_source = st.sidebar.radio("Choose video source:", ("Webcam", "Upload File"))

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

# Open video source
cap = cv2.VideoCapture(video_path)

# Streamlit video display
stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Convert BGR (OpenCV) to RGB (Streamlit)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display frame in Streamlit
    stframe.image(annotated_frame, channels="RGB")

cap.release()
