"""Streamlit app for detecting common animals in images, webcam photos, and videos."""

from __future__ import annotations

import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer


ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "ghar",
    "bison",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def get_animal_ids(model: YOLO) -> list[int]:
    return [
        class_id
        for class_id, name in model.names.items()
        if name.lower() in ANIMAL_CLASSES
    ]


def draw_status_overlay(image, text: str):
    """Draw the status text over a high-contrast bar so it stays visible."""
    height, width = image.shape[:2]
    font_scale = min(0.65, max(0.4, width / 900))
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    bar_height = text_height + baseline + 14
    cv2.rectangle(
        image,
        (0, 0),
        (min(width, text_width + 20), min(height, bar_height)),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (10, min(height - 1, text_height + 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


def record_detections(stats: dict, result, model: YOLO) -> None:
    if result.boxes is None:
        return
    for box in result.boxes:
        class_id = int(box.cls.item())
        name = model.names[class_id]
        score = float(box.conf.item())
        item = stats.setdefault(
            name,
            {
                "detections": 0,
                "confidence_sum": 0.0,
            },
        )
        item["detections"] += 1
        item["confidence_sum"] += score


def summarize_sequence(
    stats: dict,
    duration_seconds: float,
    source_name: str,
) -> str:
    duration_text = f"{duration_seconds:.1f} seconds" if duration_seconds > 0 else "an unknown duration"
    if not stats:
        return (
            f"No supported animals were detected in the {source_name}. "
            f"Analysis duration: {duration_text}."
        )

    classes = ", ".join(sorted(stats))
    detection_count = sum(item["detections"] for item in stats.values())
    confidence_sum = sum(item["confidence_sum"] for item in stats.values())
    average_confidence = 100 * confidence_sum / max(1, detection_count)
    return (
        f"Animals detected: {classes}. "
        f"Average confidence: {average_confidence:.1f}%. "
        f"Analysis duration: {duration_text}."
    )


class AnimalVideoProcessor(VideoProcessorBase):
    """Run OpenCV-backed YOLO detection on live webcam frames."""

    def __init__(self, model: YOLO, animal_ids: list[int], confidence: float):
        self.model = model
        self.animal_ids = animal_ids
        self.confidence = confidence
        self.frame_number = 0
        self.started_at = time.monotonic()
        self.stats: dict = {}
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        result = self.model.predict(
            source=image,
            classes=self.animal_ids,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )[0]
        with self.lock:
            self.frame_number += 1
            frame_number = self.frame_number
            record_detections(self.stats, result, self.model)
        annotated = draw_status_overlay(
            result.plot(),
            f"Threshold: {self.confidence:.0%} | Frame: {frame_number}",
        )
        return type(frame).from_ndarray(annotated, format="bgr24")

    def analysis_snapshot(self) -> tuple[int, float, dict]:
        with self.lock:
            copied_stats = {name: values.copy() for name, values in self.stats.items()}
            frame_number = self.frame_number
        return frame_number, max(0.0, time.monotonic() - self.started_at), copied_stats


def show_detections(result, model: YOLO) -> None:
    if result.boxes is None or len(result.boxes) == 0:
        st.info("No supported animals detected.")
        return

    rows = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        rows.append(
            {
                "Animal": model.names[class_id],
                "Confidence": f"{confidence:.1%}",
                "Box (x, y, width, height)": f"{x1}, {y1}, {x2 - x1}, {y2 - y1}",
            }
        )
    st.success(f"Detected {len(rows)} animal(s).")
    st.dataframe(rows, width="stretch", hide_index=True)


def detect_still_image(image: Image.Image, model: YOLO, animal_ids: list[int], confidence: float):
    result = model.predict(
        source=image,
        classes=animal_ids,
        conf=confidence,
        device="cpu",
        verbose=False,
    )[0]
    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detection result", width="stretch")
    st.subheader("Image analysis summary")
    if result.boxes is None or len(result.boxes) == 0:
        st.info("No supported animals were detected in this image.")
        return

    counts = Counter(model.names[int(class_id)] for class_id in result.boxes.cls.tolist())
    average_confidence = sum(float(score) for score in result.boxes.conf.tolist()) / len(result.boxes)
    animal_counts = ", ".join(
        f"{count} {animal if animal == 'sheep' else animal + ('s' if count != 1 else '')}"
        for animal, count in counts.most_common()
    )
    st.info(f"Detected {animal_counts}. Average confidence: {average_confidence:.1%}.")
    show_detections(result, model)


def detect_video(video_bytes: bytes, suffix: str, model: YOLO, animal_ids: list[int], confidence: float):
    input_path = None
    output_path = None
    capture = None
    writer = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_file:
            input_file.write(video_bytes)
            input_path = input_file.name

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            st.error("Could not open this video. Try an MP4 file encoded with H.264 or MPEG-4.")
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            st.error("The video has invalid dimensions.")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            output_path = output_file.name

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            st.error("Could not create the annotated video on this system.")
            return

        progress = st.progress(0, text="Processing video...")
        st.caption("Live annotated preview updates as each frame is processed.")
        live_preview = st.empty()
        stats: dict = {}
        frame_number = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(
                source=frame,
                classes=animal_ids,
                conf=confidence,
                device="cpu",
                verbose=False,
            )[0]
            frame_number += 1
            record_detections(stats, result, model)
            frame_total = frame_count if frame_count > 0 else "?"
            annotated = draw_status_overlay(
                result.plot(),
                f"Threshold: {confidence:.0%} | Frame: {frame_number}/{frame_total}",
            )
            writer.write(annotated)
            live_preview.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Live video analysis",
                width="stretch",
            )

            if frame_count > 0:
                progress.progress(
                    min(frame_number / frame_count, 1.0),
                    text=f"Analyzing video ({frame_number / frame_count:.0%})...",
                )

        progress.empty()
        live_preview.empty()
        if frame_number == 0:
            st.error("No readable frames were found in this video.")
            return

        with open(output_path, "rb") as output_file:
            annotated_video = output_file.read()
        st.subheader("Annotated video")
        st.video(annotated_video)
        st.download_button(
            "Download annotated video",
            data=annotated_video,
            file_name="detected_animals.mp4",
            mime="video/mp4",
        )

        st.subheader("Video analysis summary")
        st.info(summarize_sequence(stats, frame_number / fps, "video"))
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        for temporary_path in (input_path, output_path):
            if temporary_path and Path(temporary_path).exists():
                Path(temporary_path).unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="Animal Detection", page_icon="🐾", layout="wide")
    st.title("🐾 Animal Detection")
    st.write("Detect birds, cats, dogs, horses, sheep, cows, elephants, bears, zebras, and giraffes.")
    st.caption("Use live webcam detection or upload an MP4 to detect animals throughout the video.")

    with st.sidebar:
        st.header("Settings")
        confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)
        model_path = st.text_input("YOLO model", value="yolov8n.pt")
        st.caption("The model weights download automatically the first time.")

    try:
        model = load_model(model_path)
    except Exception as error:
        st.error(f"Could not load the YOLO model: {error}")
        st.stop()
    animal_ids = get_animal_ids(model)

    image_tab, camera_tab, video_tab = st.tabs(["Image upload", "Live webcam", "MP4 video"])
    with image_tab:
        image_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        if image_file is not None:
            detect_still_image(Image.open(image_file).convert("RGB"), model, animal_ids, confidence)

    with camera_tab:
        st.write("Press **START** and allow camera access to begin live detection. Press **STOP** when finished.")
        st.caption(f"Live frame overlay: Threshold {confidence:.0%} | Frame number")
        webcam = webrtc_streamer(
            key="animal-detection-webcam",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration={"iceServers": []},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: AnimalVideoProcessor(model, animal_ids, confidence),
            async_processing=True,
        )
        processor = webcam.video_processor
        if processor is not None:
            processor.confidence = confidence
            st.session_state["animal_webcam_processor"] = processor
        else:
            processor = st.session_state.get("animal_webcam_processor")

        if processor is not None:
            processed_frames, elapsed_seconds, webcam_stats = processor.analysis_snapshot()
            if getattr(webcam.state, "playing", False):
                st.caption("Live analysis is running. Press STOP to see the session summary.")
            elif processed_frames:
                st.subheader("Webcam analysis summary")
                st.info(
                    summarize_sequence(
                        webcam_stats,
                        elapsed_seconds,
                        "webcam session",
                    )
                )

    with video_tab:
        video_file = st.file_uploader(
            "Choose an MP4 video",
            type=["mp4", "mov", "avi", "mkv"],
            key="video_upload",
        )
        if video_file is not None and st.button("Detect animals in video", type="primary"):
            suffix = Path(video_file.name).suffix or ".mp4"
            detect_video(video_file.getvalue(), suffix, model, animal_ids, confidence)


if __name__ == "__main__":
    main()
