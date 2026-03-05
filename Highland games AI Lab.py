import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import openai

# --- Mediapipe Tasks API ---
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# -----------------------------
# 1. FRONTEND CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("🏴 Highland Games AI Performance Lab")

st.markdown("""
Upload a throw video (mp4 or mov), select your event, and get AI feedback
with skeleton overlay and coaching guidance.
""")

# Event selection
event = st.selectbox("Select Event", ["Wob", "WFD", "Hammer", "Stones", "Caber", "Sheaf"])

# -----------------------------
# 2. CHATGPT CONFIG
# -----------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
if not openai_api_key:
    st.warning("OpenAI API key not set. AI feedback will be unavailable.")

# -----------------------------
# 3. VIDEO UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Your Throw", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.video(video_path)  # Display original video

    # -----------------------------
    # 4. INIT MEDIAPIPE POSE (CPU SAFE)
    # -----------------------------
    MODEL_PATH = "pose_landmarker_lite.task"  # Ensure this file exists
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    pose_landmarker = PoseLandmarker(
        base_options=base_options,
        running_mode=RunningMode.VIDEO
    )

    # -----------------------------
    # 5. PROCESS VIDEO & OVERLAY SKELETON
    # -----------------------------
    processed_frames = []
    peak_hip_angle = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # PoseLandmarker detection
        try:
            result = pose_landmarker.detect_for_video(
                image=vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb),
                timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
            )
        except Exception as e:
            st.error(f"Pose detection error: {e}")
            break

        # Draw skeleton
        if result.pose_landmarks:
            for landmark in result.pose_landmarks.landmark:
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Compute simple peak hip angle example
            hip_angle = np.random.randint(90, 180)  # Replace with actual calculation
            if hip_angle > peak_hip_angle:
                peak_hip_angle = hip_angle

        processed_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Display skeleton overlay video (as GIF)
    if processed_frames:
        st.image(processed_frames, caption="Skeleton Overlay", width=640)

    # -----------------------------
    # 6. COACHING PANEL & AI FEEDBACK
    # -----------------------------
    if openai_api_key:
        st.subheader("AI Coaching Feedback")
        try:
            feedback_prompt = (
                f"Analyze this throw video for event {event}. "
                f"Peak hip angle: {peak_hip_angle}. "
                f"Give concise coaching advice and tips for improvement."
            )
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": feedback_prompt}],
                api_key=openai_api_key
            )
            advice = response.choices[0].message.content
            st.markdown(advice)
        except Exception as e:
            st.error(f"AI feedback error: {e}")
    else:
        st.info("AI feedback unavailable without OpenAI API key.")

    # -----------------------------
    # 7. PERFORMANCE METRICS
    # -----------------------------
    st.subheader("Throw Metrics")
    st.metric("Peak Hip Angle", f"{peak_hip_angle}°")

else:
    st.info("Upload a throw video to get started.")
