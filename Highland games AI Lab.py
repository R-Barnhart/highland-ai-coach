import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
import openai
import time

# -----------------------------
# MEDIA PIPE IMPORTS (TASKS API)
# -----------------------------
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# -----------------------------
# FRONTEND CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("🏴 Highland Games AI Performance Lab")
st.write(
    """
Upload your throw video, select the event, and get AI-driven feedback and performance metrics.
""")

# -----------------------------
# USER CONFIGURATION
# -----------------------------
event = st.selectbox(
    "Select Event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
openai.api_key = openai_api_key

uploaded_file = st.file_uploader("Upload Your Throw Video", type=["mp4", "mov"])

# -----------------------------
# MEDIAPIPE SETUP (CPU SAFE)
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"  # Ensure this file exists in project
base_options = BaseOptions(model_asset_path=MODEL_PATH)

pose_options = PoseLandmarker.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# -----------------------------
# PROCESS VIDEO
# -----------------------------
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Prepare video output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (frame_width, frame_height))

    # Video display placeholder
    video_placeholder = st.empty()
    ai_feedback = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # PoseLandmarker detection
        try:
            result = pose_landmarker.detect_for_video(
                frame_rgb,
                timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
            )
        except Exception:
            result = None

        # Draw skeleton overlay if landmarks detected
        if result and result.pose_landmarks:
            landmarks = result.pose_landmarks
            for i in range(len(landmarks)):
                x = int(landmarks[i].x * frame_width)
                y = int(landmarks[i].y * frame_height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections manually (simplified)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 7),  # Right arm
                (0, 4), (4, 5), (5, 6), (6, 8),  # Left arm
                (0, 9), (9, 10), (10, 11), (11, 12)  # Torso / legs
            ]
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = (int(landmarks[start_idx].x * frame_width),
                             int(landmarks[start_idx].y * frame_height))
                    end = (int(landmarks[end_idx].x * frame_width),
                           int(landmarks[end_idx].y * frame_height))
                    cv2.line(frame, start, end, (0, 255, 255), 2)

        out.write(frame)

        # Update video frame display (Streamlit works best with RGB)
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    out.release()

    st.success("✅ Video processed with skeleton overlay.")
    st.video(temp_out.name)

    # -----------------------------
    # AI FEEDBACK
    # -----------------------------
    if openai_api_key:
        st.subheader("AI Coaching Feedback")
        feedback_prompt = f"""
        You are a Highland Games AI coach.
        Analyze this throw for {event}.
        Provide feedback on technique, posture, and power.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": feedback_prompt}]
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI feedback unavailable: {e}")
    else:
        st.warning("Enter your OpenAI API key to get AI feedback.")
