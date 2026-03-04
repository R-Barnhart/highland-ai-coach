import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from fpdf import FPDF
from datetime import datetime
import mediapipe as mp

# --- Mediapipe Tasks API ---
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions


# ----------------------------
# 1. PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Highland Games AI Lab",
    layout="wide",
    page_icon="🛡️"
)

st.markdown("""
<style>
.main { background-color: #0E1117; color: #FFFFFF; }
[data-testid="stMetricValue"] { color: #00FFCC !important; font-size: 42px; font-weight: bold; }
.stButton>button { width: 100%; background-color: #FF4B4B; color: white; font-weight: bold; border: none; }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# 2. EVENT CONFIG
# ----------------------------
EVENT_PROFILES = {
    "Hammer": {"ideal": (38, 44), "tip": "Keep arms long and maximize orbit."},
    "WFD": {"ideal": (35, 42), "tip": "Drive through the trig with chest high."},
    "WOB": {"ideal": (75, 88), "tip": "Stay vertical and finish tall."},
    "Caber": {"ideal": (80, 95), "tip": "Run tall and extend through hips."},
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# ----------------------------
# 3. PDF GENERATOR
# ----------------------------
def create_pdf(event, angle, status, tip):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, "Highland Games Performance Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Event: {event}", ln=True)
    pdf.cell(0, 10, f"Peak Hip Angle: {int(angle)} degrees", ln=True)
    pdf.cell(0, 10, f"Status: {status}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Coach Feedback: {tip}")
    return pdf.output(dest="S").encode("latin-1")


# ----------------------------
# 4. SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("Coach Panel")
    event_choice = st.selectbox("Select Event", list(EVENT_PROFILES.keys()))
    playback_speed = st.slider("Playback Speed", 0.25, 1.5, 1.0)


# ----------------------------
# 5. INIT POSE LANDMARKER
# ----------------------------
MODEL_PATH = "pose_landmarker_lite.task"

base_options = BaseOptions(model_asset_path=MODEL_PATH)

pose_options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

pose_landmarker = PoseLandmarker.create_from_options(pose_options)


# ----------------------------
# 6. MAIN APP
# ----------------------------
st.title("Highland Games AI Performance Lab")

uploaded_file = st.file_uploader("Upload Throw Video", type=["mp4", "mov"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    col_video, col_data = st.columns([2, 1])

    with col_video:
        if st.button("Analyze Throw"):
            cap = cv2.VideoCapture(temp_file.name)
            video_placeholder = st.empty()

            peak_angle = 0
            peak_frame = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ✅ REQUIRED mp.Image WRAP (FIX)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )

                result = pose_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC))
                )

                if result.pose_landmarks:
                    landmarks = [(lm.x, lm.y) for lm in result.pose_landmarks[0]]
                    h, w, _ = frame.shape

                    # Draw connections
                    for connection in PoseLandmarker.POSE_CONNECTIONS:
                        start = landmarks[connection[0]]
                        end = landmarks[connection[1]]

                        cv2.line(
                            frame,
                            (int(start[0] * w), int(start[1] * h)),
                            (int(end[0] * w), int(end[1] * h)),
                            (0, 255, 0),
                            2
                        )

                    # Right hip angle (shoulder-hip-knee)
                    shoulder = landmarks[12]
                    hip = landmarks[24]
                    knee = landmarks[26]

                    angle = calculate_angle(shoulder, hip, knee)

                    if angle > peak_angle:
                        peak_angle = angle
                        peak_frame = frame.copy()

                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                time.sleep(0.03 / playback_speed)

            cap.release()

    with col_data:
        if peak_angle > 0:
            st.subheader("Results")
            st.metric("Peak Hip Angle", f"{int(peak_angle)}°")

            low, high = EVENT_PROFILES[event_choice]["ideal"]
            status = "OPTIMAL" if low <= peak_angle <= high else "ADJUSTMENT NEEDED"

            st.write(f"Status: {status}")
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
                "Download Performance Report",
                pdf_bytes,
                file_name="highland_report.pdf",
                mime="application/pdf"
            )

