import streamlit as st
import cv2
import numpy as np
import tempfile
from fpdf import FPDF
from datetime import datetime
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions


st.set_page_config(page_title="Highland Games AI Lab", layout="wide")
st.title("Highland Games AI Performance Lab")


POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),
    (11,12),
    (23,24),
    (11,23),(12,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

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


with st.sidebar:
    event_choice = st.selectbox("Select Event", list(EVENT_PROFILES.keys()))


MODEL_PATH = "pose_landmarker_lite.task"
base_options = BaseOptions(model_asset_path=MODEL_PATH)
pose_options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)


uploaded_file = st.file_uploader("Upload Throw Video", type=["mp4", "mov"])

if uploaded_file and st.button("Analyze Throw"):

    input_temp = tempfile.NamedTemporaryFile(delete=False)
    input_temp.write(uploaded_file.read())

    cap = cv2.VideoCapture(input_temp.name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = "processed_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    peak_angle = 0
    peak_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            for connection in POSE_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                cv2.line(frame,
                         (int(start[0]*w), int(start[1]*h)),
                         (int(end[0]*w), int(end[1]*h)),
                         (0,255,0), 2)

            shoulder = landmarks[12]
            hip = landmarks[24]
            knee = landmarks[26]

            angle = calculate_angle(shoulder, hip, knee)

            if angle > peak_angle:
                peak_angle = angle
                peak_frame = frame.copy()

        out.write(frame)

    cap.release()
    out.release()

    st.success("Analysis Complete")

    st.video(output_path)

    if peak_angle > 0:
        low, high = EVENT_PROFILES[event_choice]["ideal"]
        status = "OPTIMAL" if low <= peak_angle <= high else "ADJUSTMENT NEEDED"

        st.metric("Peak Hip Angle", f"{int(peak_angle)}°")
        st.write(status)
        st.info(EVENT_PROFILES[event_choice]["tip"])

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
