import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from fpdf import FPDF
from datetime import datetime

# Mediapipe Tasks API
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Highland Games AI Lab", layout="wide", page_icon="🛡️")

# -------------------------------
# MODEL SETUP
# -------------------------------
MODEL_PATH = "pose_landmarker_lite.task"

base_options = BaseOptions(model_asset_path=MODEL_PATH)

pose_options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)

# -------------------------------
# EVENT PROFILES
# -------------------------------
EVENT_PROFILES = {
    "Hammer (Light/Heavy)": {"ideal": (38, 44), "tip": "Maximize orbit! Keep arms extended."},
    "WOB (Weight for Height)": {"ideal": (75, 88), "tip": "Drive vertically. Chest tall."},
    "WFD (Weight for Distance)": {"ideal": (35, 42), "tip": "Drive through the trig."},
    "Sheaf Toss": {"ideal": (65, 82), "tip": "Hinge and snap the hips."},
    "Caber Toss": {"ideal": (80, 95), "tip": "Stay tall during transition."},
    "Open Stone": {"ideal": (37, 43), "tip": "Explosive weight transfer."},
    "Braemar Stone": {"ideal": (39, 45), "tip": "Strong leg drive."}
}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def create_pdf(event, angle, status, tip):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 10, txt="Highland Games Performance Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Event: {event}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Peak Hip Extension: {int(angle)} degrees", ln=True)
    pdf.cell(200, 10, txt=f"Status: {status}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 12)
    pdf.multi_cell(0, 10, txt=f"Coach Feedback: {tip}")
    return pdf.output(dest='S').encode('latin-1')


# -------------------------------
# UI
# -------------------------------
with st.sidebar:
    st.title("🛡️ Coach Panel")
    event_choice = st.selectbox("Select Event", list(EVENT_PROFILES.keys()))
    play_speed = st.slider("Playback Speed", 0.1, 1.0, 1.0)

st.title("Highland Games AI Performance Lab")

uploaded_file = st.file_uploader("Upload Your Throw", type=["mp4", "mov"])

# -------------------------------
# VIDEO PROCESSING
# -------------------------------
if uploaded_file:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    col_vid, col_data = st.columns([2, 1])

    with col_vid:
        if st.button("🚀 Analyze Form"):

            cap = cv2.VideoCapture(temp_file.name)
            st_frame = st.empty()

            peak_angle = 0
            peak_frame = None
            timestamp_ms = 0

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = pose_landmarker.detect_for_video(rgb, int(timestamp_ms))
                timestamp_ms += 1000 / fps

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    h, w, _ = frame.shape

                    # Draw landmarks
                    for lm in landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    # Calculate hip angle (shoulder-hip-knee)
                    shoulder = (landmarks[12].x, landmarks[12].y)
                    hip = (landmarks[24].x, landmarks[24].y)
                    knee = (landmarks[26].x, landmarks[26].y)

                    angle = calculate_angle(shoulder, hip, knee)

                    cv2.putText(frame, f"Hip Angle: {int(angle)}",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2)

                    if angle > peak_angle:
                        peak_angle = angle
                        peak_frame = frame.copy()

                st_frame.image(frame, channels="BGR", use_container_width=True)
                time.sleep(0.03 / play_speed)

            cap.release()

            # ---------------------------
            # RESULTS PANEL
            # ---------------------------
            with col_data:
                st.subheader("Session Results")
                st.metric("Peak Angle", f"{int(peak_angle)}°")

                low, high = EVENT_PROFILES[event_choice]["ideal"]
                status = "OPTIMAL" if low <= peak_angle <= high else "ADJUSTMENT NEEDED"

                st.write(f"**Status:** {status}")
                st.info(EVENT_PROFILES[event_choice]["tip"])

                if peak_frame is not None:
                    st.image(peak_frame, channels="BGR", caption="Peak Frame")

                pdf_bytes = create_pdf(
                    event_choice,
                    peak_angle,
                    status,
                    EVENT_PROFILES[event_choice]["tip"]
                )

                st.download_button(
                    "📥 Download Report",
                    data=pdf_bytes,
                    file_name="Highland_Report.pdf",
                    mime="application/pdf"
                )
