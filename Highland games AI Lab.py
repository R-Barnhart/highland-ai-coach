import streamlit as st
import cv2
import numpy as np
import tempfile
import time
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
# EVENT PROFILES
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
# ANGLE CALCULATION
# ---------------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ---------------------------------
# PDF REPORT
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
    play_speed = st.slider("Playback Speed", 0.25, 1.5, 1.0)

st.title("Highland Games AI Performance Lab")
uploaded_file = st.file_uploader("Upload Throw Video", type=["mp4", "mov"])

# ---------------------------------
# VIDEO PROCESSING
# ---------------------------------
if uploaded_file:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    col_video, col_data = st.columns([2, 1])

    with col_video:
        if st.button("🚀 Analyze Form"):

            cap = cv2.VideoCapture(temp_file.name)
            st_frame = st.empty()

            peak_angle = 0
            peak_frame = None
            angle_history = []

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30

            timestamp_ms = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- FIXED IMAGE WRAPPER ---
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
                    h, w, _ = frame.shape

                    # Draw skeleton connections
                    connections = mp.solutions.pose.POSE_CONNECTIONS
                    for connection in connections:
                        start = landmarks[connection[0]]
                        end = landmarks[connection[1]]
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    # Calculate hip angle
                    shoulder = (landmarks[12].x, landmarks[12].y)
                    hip = (landmarks[24].x, landmarks[24].y)
                    knee = (landmarks[26].x, landmarks[26].y)

                    angle = calculate_angle(shoulder, hip, knee)

                    # Smooth small fluctuations
                    if len(angle_history) > 5:
                        angle = np.mean(angle_history[-5:])

                    angle_history.append(angle)

                    cv2.putText(frame, f"Hip Angle: {int(angle)}°",
                                (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2)

                    if angle > peak_angle:
                        peak_angle = angle
                        peak_frame = frame.copy()

                st_frame.image(frame, channels="BGR", use_container_width=True)
                time.sleep(0.02 / play_speed)

            cap.release()

            # ---------------------------------
            # RESULTS PANEL
            # ---------------------------------
            with col_data:
                st.subheader("Session Results")
                st.metric("Peak Hip Angle", f"{int(peak_angle)}°")

                low, high = EVENT_PROFILES[event_choice]["ideal"]
                status = "OPTIMAL" if low <= peak_angle <= high else "ADJUSTMENT NEEDED"

                st.write(f"**Status:** {status}")
                st.info(EVENT_PROFILES[event_choice]["tip"])

                # Angle chart
                st.line_chart(angle_history)

                if peak_frame is not None:
                    st.image(peak_frame, channels="BGR", caption="Peak Frame")

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

