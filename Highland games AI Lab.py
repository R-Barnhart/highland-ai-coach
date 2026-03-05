import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import openai

# -----------------------------
# MediaPipe Tasks API imports
# -----------------------------
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import RunningMode

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")

# -----------------------------
# Header
# -----------------------------
st.title("🏴 Highland Games AI Lab")
st.markdown("""
Analyze your throws with AI-powered pose detection and coaching feedback.
""")

# -----------------------------
# Event Selection
# -----------------------------
event = st.selectbox(
    "Choose Your Event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

# -----------------------------
# Video Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Your Throw Video", type=["mp4", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Display uploaded video
    st.video(video_path)

# -----------------------------
# Initialize PoseLandmarker
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"
pose_options = PoseLandmarker.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# -----------------------------
# Process Video & Overlay Skeleton
# -----------------------------
if uploaded_file is not None:
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Temporary video output
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))

    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_landmarker.detect_for_video(
            rgb_frame,
            timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        # Draw skeleton
        if result.pose_landmarks:
            landmarks = result.pose_landmarks
            all_landmarks.append([(lm.x, lm.y) for lm in landmarks])

            for connection in PoseLandmarker.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                cv2.line(
                    frame,
                    (int(start.x * width), int(start.y * height)),
                    (int(end.x * width), int(end.y * height)),
                    (0, 255, 0),
                    2
                )

            for lm in landmarks:
                cv2.circle(
                    frame,
                    (int(lm.x * width), int(lm.y * height)),
                    4,
                    (0, 0, 255),
                    -1
                )

        out.write(frame)

    cap.release()
    out.release()

    # Display skeleton overlay video
    st.subheader("Skeleton Overlay Video")
    st.video(out_file.name)

    # -----------------------------
    # AI Throw Feedback
    # -----------------------------
    if all_landmarks:
        # Convert landmarks into a string for analysis
        landmarks_text = "\n".join([str(lm) for frame in all_landmarks for lm in frame])

        # Send to OpenAI GPT
        openai.api_key = st.secrets["OPENAI_API_KEY"]  # store securely
        prompt = f"""
        Analyze this Highland Games throw data (event: {event}) and give actionable feedback:
        {landmarks_text}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            feedback = response.choices[0].message.content
        except Exception as e:
            feedback = f"AI feedback unavailable: {e}"

        st.subheader("AI Coaching Feedback")
        st.write(feedback)
