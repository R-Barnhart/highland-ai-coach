import streamlit as st
from PIL import Image
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import RunningMode
import openai
import tempfile
import time

# -----------------------------
# 1. STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("🏴 Highland Games AI Performance Lab")
st.markdown(
    """
    Upload your throw video and get AI-driven feedback.
    The system supports: WOB, WFD, Hammer, Stones, Caber, Sheaf.
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 2. USER INPUT
# -----------------------------
event = st.selectbox(
    "Select your event",
    ["WOB", "WFD", "Hammer", "Stones", "Caber", "Sheaf"]
)
video_file = st.file_uploader("Upload Your Throw Video", type=["mp4", "mov"])

# -----------------------------
# 3. OPENAI SETUP
# -----------------------------
# Make sure you have your OpenAI key in your environment or paste it here
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_KEY

# -----------------------------
# 4. MEDIAPIPE POSE LANDMARKER (CPU SAFE)
# -----------------------------
MODEL_PATH = "pose_landmarker_lite.task"  # Must exist in project folder

base_options = BaseOptions(model_asset_path=MODEL_PATH)
pose_options = PoseLandmarker.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# -----------------------------
# 5. PROCESS VIDEO
# -----------------------------
if video_file:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Video display
    stframe = st.empty()

    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Pose detection ---
        result = pose_landmarker.detect_for_video(
            rgb_frame,
            timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        if result.pose_landmarks:
            landmarks = result.pose_landmarks

            # Draw skeleton overlay
            for connection in PoseLandmarker.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, start, end, (0, 255, 0), 2)

            # Collect data for AI feedback
            all_landmarks.append([(lm.x, lm.y, lm.z) for lm in landmarks])

        stframe.image(frame, channels="BGR")
        time.sleep(0.01)

    cap.release()

    # -----------------------------
    # 6. AI FEEDBACK
    # -----------------------------
    if OPENAI_KEY:
        st.subheader("AI Coaching Feedback")
        try:
            prompt = f"""
            Analyze this {event} throw based on landmarks data.
            Data: {all_landmarks[:10]} ... (truncated for brevity)
            Provide specific coaching feedback and tips for improvement.
            """
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            st.write(response['choices'][0]['message']['content'])
        except Exception as e:
            st.error(f"AI feedback unavailable: {e}")
    else:
        st.warning("No OpenAI API key provided. AI feedback unavailable.")
