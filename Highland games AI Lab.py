# ============================================
# HIGHLAND GAMES AI COACH
# Professional Version
# ============================================

import os

# ---- FORCE CPU MODE (Fix EGL crash) ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from openai import OpenAI

# ============================================
# STREAMLIT PAGE
# ============================================

st.set_page_config(
    page_title="Highland Games AI Coach",
    layout="wide"
)

st.title("🏴 Highland Games AI Coach")

st.markdown("Upload a throwing video to receive AI coaching feedback.")

# ============================================
# OPENAI CLIENT
# ============================================

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except:
    client = None

# ============================================
# EVENT SELECTOR
# ============================================

event = st.selectbox(
    "Select Event",
    [
        "Weight Over Bar (WOB)",
        "Weight For Distance (WFD)",
        "Hammer Throw",
        "Stone Throw",
        "Caber Toss",
        "Sheaf Toss"
    ]
)

uploaded_video = st.file_uploader("Upload Throw Video", type=["mp4", "mov", "avi"])

# ============================================
# MEDIAPIPE SETUP (CPU SAFE)
# ============================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# POSE DRAWING STYLE
# ============================================

skeleton_style = mp_drawing.DrawingSpec(
    color=(0,255,0),
    thickness=3,
    circle_radius=3
)

joint_style = mp_drawing.DrawingSpec(
    color=(255,0,0),
    thickness=3,
    circle_radius=4
)

# ============================================
# VIDEO PROCESSING
# ============================================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    out = cv2.VideoWriter(
        output_file.name,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height)
    )

    frame_count = 0
    pose_points = []

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                skeleton_style,
                joint_style
            )

            pose_points.append(str(results.pose_landmarks))

        out.write(frame)

    cap.release()
    out.release()

    return output_file.name, frame_count, pose_points


# ============================================
# AI COACHING
# ============================================

def ai_coach(event, frames):

    if client is None:
        return "⚠️ AI feedback unavailable. Add OPENAI_API_KEY to Streamlit secrets."

    prompt = f"""
You are a professional Highland Games throwing coach.

Analyze the athlete motion data and provide coaching advice.

Event: {event}

Focus on:

- footwork
- hip drive
- timing
- release angle
- balance
- power transfer

Provide:
1. What the athlete did well
2. Technical mistakes
3. Specific corrections
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are an elite Highland Games coach."},
                {"role":"user","content":prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI error: {str(e)}"

# ============================================
# RUN ANALYSIS
# ============================================

if uploaded_video:

    st.subheader("Original Video")
    st.video(uploaded_video)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

    if st.button("Analyze Throw"):

        with st.spinner("Analyzing throw mechanics..."):

            overlay_video, frame_count, frames = process_video(video_path)

            feedback = ai_coach(event, frames)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Skeleton Overlay")
            st.video(overlay_video)

        with col2:
            st.subheader("AI Coaching Feedback")
            st.write(feedback)

        st.success(f"Processed {frame_count} frames successfully.")
