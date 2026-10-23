import cv2
from ultralytics import YOLO

# Load YOLOv8 model (download yolov8n.pt from Ultralytics)
model = YOLO("models/yolov8n.pt")

# Load video file or webcam (0 for webcam)
cap = cv2.VideoCapture("data/sample.mp4")

# Define video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("outputs/detected_output.avi", fourcc, 20.0, 
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Video Detection", annotated_frame)

    # Save frame
    out.write(annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
