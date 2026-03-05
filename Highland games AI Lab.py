import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
import openai
import tempfile
import time

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")

# -----------------------------
# Sidebar options
# -----------------------------
st.sidebar.title("Highland Games Settings")
event_option = st.sidebar.selectbox(
    "Select Event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

# AI feedback key
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password"
)
openai.api_key = openai_api_key

# -----------------------------
# Video upload
# -----------------------------
uploaded_video = st.file_uploader(
    "Upload Throw Video", type=["mp4", "mov", "avi"]
)

# -----------------------------
# PoseLandmarker Setup (CPU safe)
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"  # Must exist in project folder

if not st.session_state.get("pose_model"):
    # Load model once
    def load_model():
        if not MODEL_PATH:
            st.error("Pose model file missing. Add pose_landmarker_lite.task")
            st.stop()
        return PoseLandmarker.create_from_model_path(MODEL_PATH)
    st.session_state.pose_model = load_model()

pose_model = st.session_state.pose_model

# -----------------------------
# Skeleton overlay function
# -----------------------------
def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # Manual PoseLandmarker connections
    connections = [
        (0,1),(1,2),(2,3),(3,7),       # Right arm
        (0,4),(4,5),(5,6),(6,8),       # Left arm
        (9,10),                         # Hips
        (11,12),(12,14),(14,16),        # Right leg
        (11,13),(13,15),(15,17)         # Left leg
    ]

    for start, end in connections:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (0,255,0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0,0,255), -1)

    return frame

# -----------------------------
# Video processing & skeleton overlay
# -----------------------------
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    prev_timestamp = -1  # for monotonic timestamp

    peak_hip_angle = 0  # placeholder metric

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection with monotonic timestamp
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= prev_timestamp:
            timestamp_ms = prev_timestamp + 1
        prev_timestamp = timestamp_ms

        # Create mediapipe Image from numpy array
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame_rgb)

        # Detect pose
        result = pose_model.detect_for_video(mp_image, timestamp_ms)

        # Draw skeleton if landmarks detected
        if result.pose_landmarks:
            frame = draw_skeleton(frame, result.pose_landmarks)

        # Display frame
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

    # -----------------------------
    # AI Feedback Panel
    # -----------------------------
    if openai_api_key:
        st.subheader("AI Feedback on Throw")
        try:
            # Example: simple feedback request
            feedback_prompt = f"Analyze this throw video for {event_option} and provide coaching advice."
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"user", "content": feedback_prompt}],
                max_tokens=200
            )
            ai_feedback = response.choices[0].message.content
            st.text(ai_feedback)
        except Exception as e:
            st.error(f"AI feedback unavailable: {e}")
    else:
        st.info("Enter your OpenAI API key to get AI feedback.")




