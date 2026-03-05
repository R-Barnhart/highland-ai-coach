import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import requests
import json

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide")

st.title("Highland Games AI Coach")
st.write("Upload a throw video and receive AI coaching feedback.")

# -----------------------------
# EVENT SELECTOR
# -----------------------------
event = st.selectbox(
    "Select Event",
    ["Weight Over Bar (WOB)", "Weight For Distance (WFD)", "Hammer Throw", "Stone Put", "Caber Toss", "Sheaf Toss"]
)

# -----------------------------
# COACHING PANEL
# -----------------------------
st.sidebar.title("Coaching Panel")

st.sidebar.write("This panel will show AI coaching feedback.")

coaching_output = st.sidebar.empty()

# -----------------------------
# OPENAI SETTINGS
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

def get_ai_feedback(event, metrics):

    prompt = f"""
You are a professional Highland Games coach.

Event: {event}

Athlete metrics from pose analysis:
{metrics}

Provide clear coaching advice including:
- What the athlete did well
- What technical problems exist
- How to fix them
- Drills to improve

Keep the tone like a professional throwing coach.
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]

    return "AI feedback unavailable."

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# VIDEO UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("Upload Throw Video", type=["mp4","mov","avi"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.subheader("Original Video")
    st.video(video_path)

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(
        output_file.name,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    hip_angles = []
    shoulder_rotation = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            lm = results.pose_landmarks.landmark

            shoulder = np.array([lm[11].x, lm[11].y])
            hip = np.array([lm[23].x, lm[23].y])
            knee = np.array([lm[25].x, lm[25].y])

            a = shoulder - hip
            b = knee - hip

            cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angle = np.degrees(np.arccos(cosine))

            hip_angles.append(angle)

        out.write(frame)

        progress.progress(frame_count / total_frames)

    cap.release()
    out.release()

    st.subheader("Skeleton Overlay Video")
    st.video(output_file.name)

# -----------------------------
# ANALYSIS RESULTS
# -----------------------------
    if len(hip_angles) > 0:

        peak_hip = max(hip_angles)
        avg_hip = np.mean(hip_angles)

        st.subheader("Throw Metrics")

        col1, col2 = st.columns(2)

        col1.metric("Peak Hip Angle", f"{peak_hip:.1f}°")
        col2.metric("Average Hip Angle", f"{avg_hip:.1f}°")

        st.line_chart(hip_angles)

        metrics = {
            "peak_hip_angle": float(peak_hip),
            "average_hip_angle": float(avg_hip),
            "frames_analyzed": len(hip_angles)
        }

        st.subheader("AI Coaching Feedback")

        if OPENAI_API_KEY != "":
            feedback = get_ai_feedback(event, metrics)
            st.write(feedback)
            coaching_output.write(feedback)
        else:
            st.warning("Add your OpenAI API key to Streamlit secrets to enable AI coaching.")

    else:
        st.error("Pose not detected. Try a clearer video.")


