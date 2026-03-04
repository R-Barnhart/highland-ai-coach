import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
from fpdf import FPDF
from datetime import datetime
import os

# --- 1. FRONTEND CONFIGURATION ---
st.set_page_config(page_title="Highland Games AI Lab", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stMetricValue"] { color: #00FFCC !important; font-size: 42px; font-weight: bold; }
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; font-weight: bold; border: none; }
    .report-box { padding: 20px; border: 1px solid #333; border-radius: 10px; background-color: #1A1C24; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. AI & PHYSICS SETUP ---
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

EVENT_PROFILES = {
    "Hammer (Light/Heavy)": {"ideal": (38, 44), "tip": "Maximize orbit! Keep arms fully extended during the winds."},
    "WOB (Weight for Height)": {"ideal": (75, 88), "tip": "Vertical drive! Don't let the weight pull your chest down."},
    "WFD (Weight for Distance)": {"ideal": (35, 42), "tip": "Drive through the trig! Keep the chest high at release."},
    "Sheaf Toss": {"ideal": (65, 82), "tip": "Hinge and snap! Use your hips to flick the fork upward."},
    "Caber Toss": {"ideal": (80, 95), "tip": "Tall posture! Look at the horizon during the run and transition."},
    "Open Stone": {"ideal": (37, 43), "tip": "Explosive glide! Transfer weight from back to front quickly."},
    "Braemar Stone": {"ideal": (39, 45), "tip": "Leg drive! Keep the stone tucked until the final extension."}
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad*180.0/np.pi)
    return 360 - angle if angle > 180 else angle

# --- 3. PDF GENERATOR ---
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
    pdf.set_text_color(0, 128, 0) if status == "OPTIMAL" else pdf.set_text_color(255, 0, 0)
    pdf.cell(200, 10, txt=f"Status: {status}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 12)
    pdf.multi_cell(0, 10, txt=f"Coach's Feedback: {tip}")
    return pdf.output(dest='S').encode('latin-1')

# --- 4. SIDEBAR UI ---
with st.sidebar:
    st.title("🛡️ Coach's Panel")
    event_choice = st.selectbox("Select Event", list(EVENT_PROFILES.keys()))
    play_speed = st.slider("Playback Speed", 0.1, 1.0, 1.0)
    st.divider()

st.title("Highland Games AI Performance Lab")
u_user = st.file_uploader("Upload Your Throw", type=["mp4", "mov"])

# --- 5. LOAD POSE LANDMARKER MODEL ---
MODEL_PATH = "pose_landmarker_lite.task"  # Must exist in project folder
base_options = BaseOptions(model_asset_path=MODEL_PATH)
pose_options = PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
pose_landmarker = PoseLandmarker.create(pose_options)

# --- 6. PROCESS VIDEO ---
if u_user:
    t_u = tempfile.NamedTemporaryFile(delete=False)
    t_u.write(u_user.read())
    t_u.flush()

    col_vid, col_data = st.columns([2, 1])

    with col_vid:
        if st.button("🚀 Analyze Form"):
            cap = cv2.VideoCapture(t_u.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Production-safe video writer
            temp_output_path = "processed_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # cross-platform safe
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            st_vid = st.empty()
            peak_angle, peak_frame = 0, None
            timestamp_ms = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: 
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose_landmarker.detect_for_video(rgb, timestamp_ms)
                timestamp_ms += int(1000 / fps)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks
                    # Draw connections
                    for connection in mp.solutions.pose.POSE_CONNECTIONS:
                        start = landmarks[connection[0]]
                        end = landmarks[connection[1]]
                        cv2.line(frame,
                                 (int(start.x * width), int(start.y * height)),
                                 (int(end.x * width), int(end.y * height)),
                                 (0, 255, 0), 2)
                    # Calculate hip-knee-shoulder angle
                    s = [landmarks[12].x, landmarks[12].y]
                    h = [landmarks[24].x, landmarks[24].y]
                    k = [landmarks[26].x, landmarks[26].y]
                    ang = calculate_angle(s, h, k)
                    if ang > peak_angle:
                        peak_angle, peak_frame = ang, frame.copy()

                out.write(frame)
                st_vid.image(frame, channels="BGR", use_container_width=True)
                time.sleep(0.03 / play_speed)

            cap.release()
            out.release()

            # --- DATA DASHBOARD ---
            with col_data:
                st.subheader("Session Results")
                st.metric("Peak Angle", f"{int(peak_angle)}°")

                low, high = EVENT_PROFILES[event_choice]["ideal"]
                status = "OPTIMAL" if low <= peak_angle <= high else "ADJUSTMENT NEEDED"

                st.write(f"**Status:** {status}")
                st.info(f"**Tip:** {EVENT_PROFILES[event_choice]['tip']}")

                if peak_frame is not None:
                    st.image(peak_frame, channels="BGR", caption="Moment of Peak Power")

                # PDF DOWNLOAD
                pdf_bytes = create_pdf(event_choice, peak_angle, status, EVENT_PROFILES[event_choice]['tip'])
                st.download_button(
                    label="📥 Download Performance Report",
                    data=pdf_bytes,
                    file_name=f"Highland_Report_{event_choice.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

            # --- PLAY THE VIDEO ---
            st.subheader("Processed Video with Skeleton Overlay")
            st.video(temp_output_path)

