import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from fpdf import FPDF
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide", page_icon="🛡️")

# ---------------------------------
# LOAD MODEL
# ---------------------------------
MODEL_PATH = "pose_landmarker_lite.task"

base_options = BaseOptions(model_asset_path=MODEL_PATH)
pose_options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# ---------------------------------
# MANUAL POSE CONNECTIONS
# ---------------------------------
POSE_CONNECTIONS = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),
    (24,26),(26,28)
]

# ---------------------------------
# EVENTS
# ---------------------------------
EVENT_PROFILES = {
    "Hammer": {"ideal": (38, 44), "tip": "Keep arms long and maximize orbit."},
    "WOB": {"ideal": (75, 88), "tip": "Drive vertically. Chest tall."},
    "WFD": {"ideal": (35, 42), "tip": "Drive through the trig."},
    "Sheaf": {"ideal": (65, 82), "tip": "Hinge and snap hips."},
    "Caber": {"ideal": (80, 95), "tip": "Stay tall through transition."},
    "Open Stone": {"ideal": (37, 43), "tip": "Explosive weight transfer."},
    "Braemar": {"ideal": (39, 45), "tip": "Strong leg drive."}
}

# ---------------------------------
# ANGLE FUNCTION
# ---------------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ---------------------------------
# PDF
# ---------------------------------
def create_pdf(event, angle, status, tip):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 10, "Highland Games Performance Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, f"Event: {event}", ln=True)
    pdf.cell(200, 10, f"Peak Hip Angle: {int(angle)}°", ln=True)
    pdf.cell(200, 10, f"Status: {status}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Coach Feedback: {tip}")
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------
# UI
# ---------------------------------
with st.sidebar:
    st.title("🛡️ Coach Panel")
    event_choice = st.selectbox("Select Event", list(EVENT_PROFILES.keys()))

st.title("Highland Games AI Performance Lab")
uploaded_file = st.file_uploader("Upload Throw Video", type=["mp4", "mov"])

# ---------------------------------
# PROCESS VIDEO SAFELY
# ---------------------------------
if uploaded_file:

    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_file.read())
    temp_input.close()

    if st.button("🚀 Analyze Form"):

        cap = cv2.VideoCapture(temp_input.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_output_path = "processed_output.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        peak_angle = 0
        angle_history = []
        timestamp_ms = 0

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = pose_landmarker.detect_for_video(
                mp_image,
                int(timestamp_ms)
            )

            timestamp_ms += 1000 / fps

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Draw skeleton
                for connection in POSE_CONNECTIONS:
                    start = landmarks[connection[0]]
                    end = landmarks[connection[1]]

                    x1, y1 = int(start.x * width), int(start.y * height)
                    x2, y2 = int(end.x * width), int(end.y * height)

                    cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                shoulder = (landmarks[12].x, landmarks[12].y)
                hip = (landmarks[24].x, landmarks[24].y)
                knee = (landmarks[26].x, landmarks[26].y)

                angle = calculate_angle(shoulder, hip, knee)
                angle_history.append(angle)

                if angle > peak_angle:
                    peak_angle = angle

                cv2.putText(frame, f"Hip Angle: {int(angle)}°",
                            (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

            out.write(frame)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()

        progress.empty()

        # ---------------------------------
        # DISPLAY RESULTS
        # ---------------------------------
        st.subheader("Processed Video")
        st.video(temp_output_path)

        st.subheader("Session Results")
        st.metric("Peak Hip Angle", f"{int(peak_angle)}°")

        low, high = EVENT_PROFILES[event_choice]["ideal"]
        status = "OPTIMAL" if low <= peak_angle <= high else "ADJUSTMENT NEEDED"

        st.write(f"**Status:** {status}")
        st.info(EVENT_PROFILES[event_choice]["tip"])

        st.line_chart(angle_history)

        pdf_bytes = create_pdf(
            event_choice,
            peak_angle,
            status,
            EVENT_PROFILES[event_choice]["tip"]
        )

        st.download_button(
            "📥 Download Performance Report",
            data=pdf_bytes,
            file_name="Highland_Report.pdf",
            mime="application/pdf"
        )

        os.remove(temp_input.name)
