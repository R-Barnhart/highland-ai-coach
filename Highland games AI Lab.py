import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess
import base64
import io
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE  (requires mediapipe==0.10.14 in requirements.txt)
# model_complexity=1 uses the bundled lite model — no download needed
# ─────────────────────────────────────────────────────────────────────────────
import mediapipe as mp
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

LANDMARK_SPEC   = mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=3, circle_radius=5)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=3)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Highland Games AI Coach", layout="centered")

st.title("🏴 Highland Games AI Throw Coach")
st.write("Upload a throwing video to get a wireframe overlay and GPT-4o coaching feedback.")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — API key + event selector
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OpenAI API Key", type="password",
                               placeholder="sk-...",
                               help="Get yours at platform.openai.com/api-keys")
    if openai_key:
        st.success("API key saved ✓", icon="🔐")

    st.divider()
    event = st.selectbox("Highland Games Event", [
        "Auto-detect",
        "Caber Toss",
        "Scottish Hammer Throw",
        "Weight for Distance (light)",
        "Weight for Distance (heavy)",
        "Stone Put (Braemar)",
        "Stone Put (Open)",
        "Weight Over Bar",
        "Sheaf Toss",
    ])

    st.divider()
    num_frames = st.slider("Key frames sent to GPT", 4, 10, 6,
                           help="More frames = richer analysis, slightly higher cost.")

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_video = st.file_uploader("Upload Throw Video", type=["mp4", "mov", "avi", "mpeg4"])

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def reencode_h264(src: str, dst: str) -> tuple:
    """Re-encode src to H.264 mp4 so every browser can play it."""
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(candidate) or \
                subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
            ffmpeg = candidate
            break
    else:
        return False, "ffmpeg not found — add 'ffmpeg' to packages.txt"

    r = subprocess.run(
        [ffmpeg, "-y", "-i", src,
         "-vcodec", "libx264", "-preset", "fast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 500:
        return False, r.stderr[-400:] if r.stderr else "empty output"
    return True, ""


def pick_frame_indices(total: int, n: int) -> list:
    """Evenly spread n indices across the middle 90% of the video."""
    start = max(0, int(total * 0.05))
    end   = min(total - 1, int(total * 0.95))
    if start >= end:
        return list(range(min(n, total)))
    step = max(1, (end - start) // (n - 1))
    return [start + i * step for i in range(n)][:n]


def frame_to_b64(frame_rgb: np.ndarray) -> str:
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def gpt_coaching(api_key: str, frames_rgb: list, event_name: str) -> str:
    """Send annotated key frames to GPT-4o and return coaching text."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system = """You are an elite Highland Games coach and sports biomechanics expert
with 20+ years experience coaching caber toss, hammer throw, weight-for-distance,
stone put, sheaf toss, and weight-over-bar events.

You will receive sequential key frames from a throw video. Each frame has a
MediaPipe pose wireframe overlaid (green dots = joints, blue lines = skeleton).

Structure your response with these sections:
**Event Identified** — name the event based on visual cues.
**Phase Breakdown** — setup/wind-up → power phase → release → follow-through.
**Biomechanics** — hip-shoulder separation, foot placement, arm path, spine angle, head position.
**Strengths** — what the athlete is doing well (be specific, reference the skeleton).
**Top 3 Improvements** — the highest-impact coaching cues for more distance.
**Recommended Drills** — 2-3 targeted drills to fix the weaknesses identified.

Be direct, technically precise, and encouraging."""

    content = [{
        "type": "text",
        "text": (f"Analyse this **{event_name}** throw. "
                 f"Here are {len(frames_rgb)} sequential key frames with pose wireframe overlays.")
    }]
    for i, frame in enumerate(frames_rgb):
        content.append({"type": "text", "text": f"Frame {i+1} of {len(frames_rgb)}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_to_b64(frame)}", "detail": "high"}
        })

    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": content},
        ],
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_W = 720   # standard display width for the output video
DISPLAY_H = 404   # 16:9

def process_video(input_path: str, raw_out: str, key_indices: set):
    """
    Draw pose wireframe on every frame, write to raw_out.
    Collect key frames (RGB) at key_indices for GPT.
    Returns (key_frames, total_frames, landmarks_detected).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, 0, 0

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale to standard display size, keep aspect ratio
    scale  = min(DISPLAY_W / native_w, DISPLAY_H / native_h)
    out_w  = int(native_w * scale) // 2 * 2   # keep even
    out_h  = int(native_h * scale) // 2 * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (out_w, out_h))

    key_frames         = []
    frame_index        = 0
    landmarks_detected = 0
    bar = st.progress(0, text="Processing pose wireframe…")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose.process(rgb)
            annotated = frame.copy()

            if results.pose_landmarks:
                landmarks_detected += 1
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=LANDMARK_SPEC,
                    connection_drawing_spec=CONNECTION_SPEC,
                )

            # Subtle frame counter in corner
            cv2.putText(annotated, f"{frame_index}/{total}",
                        (8, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 160), 1)

            writer.write(annotated)

            # Collect key frame for GPT (RGB, annotated)
            if frame_index in key_indices:
                key_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            frame_index += 1
            if total > 0:
                bar.progress(min(frame_index / total, 1.0),
                             text=f"Frame {frame_index} / {total}")

    cap.release()
    writer.release()
    bar.empty()

    return key_frames, frame_index, landmarks_detected


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_video is not None:

    if not openai_key:
        st.warning("⚠️  Enter your OpenAI API key in the sidebar to enable GPT-4o coaching.")

    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.read())
        input_path = tmp.name

    raw_path = input_path.replace(suffix, "_raw.mp4")
    web_path = input_path.replace(suffix, "_web.mp4")

    # Quick peek at total frames to choose key indices
    _cap   = cv2.VideoCapture(input_path)
    _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    key_idx_list = pick_frame_indices(_total, num_frames)
    key_idx_set  = set(key_idx_list)

    # ── Step 1: pose wireframe ────────────────────────────────────────────────
    key_frames, total_frames, lm_count = process_video(input_path, raw_path, key_idx_set)

    if key_frames is None:
        st.error("Could not open video file. Please try a different file.")
        st.stop()

    detect_pct = int(lm_count / max(total_frames, 1) * 100)
    if detect_pct >= 60:
        st.success(f"✅  Pose detected in {detect_pct}% of {total_frames} frames.")
    elif detect_pct > 0:
        st.warning(f"⚠️  Pose only detected in {detect_pct}% of frames. "
                   "Try better lighting or ensure the full body is visible.")
    else:
        st.error("❌  No pose landmarks detected. Ensure the athlete is clearly visible "
                 "and the full body is in frame.")

    # ── Step 2: re-encode for browser ────────────────────────────────────────
    with st.spinner("Preparing video…"):
        ok, enc_err = reencode_h264(raw_path, web_path)

    play_path = web_path if ok else raw_path
    if not ok:
        st.warning(f"Re-encode failed: `{enc_err}` — add `ffmpeg` to packages.txt")

    st.subheader("🎬 Wireframe Overlay")
    with open(play_path, "rb") as f:
        st.video(f.read())

    # ── Step 3: key frame strip ───────────────────────────────────────────────
    if key_frames:
        st.subheader("🖼️ Key Frames Sent to GPT-4o")
        cols = st.columns(len(key_frames))
        for col, (frame, idx) in zip(cols, zip(key_frames, key_idx_list)):
            with col:
                st.image(frame, caption=f"Frame {idx}", use_container_width=True)

    # ── Step 4: GPT-4o coaching ───────────────────────────────────────────────
    if openai_key and key_frames:
        st.subheader("🏆 GPT-4o Coaching Feedback")
        event_label = event if event != "Auto-detect" else "Highland Games throw"

        with st.spinner("GPT-4o is analysing your throw…"):
            try:
                feedback = gpt_coaching(openai_key, key_frames, event_label)
                st.markdown(feedback)

                st.download_button(
                    "📥 Download Coaching Report",
                    data=feedback,
                    file_name="coaching_report.txt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"GPT-4o error: {e}")

    elif not openai_key:
        st.info("Add your OpenAI API key in the sidebar to get GPT-4o coaching feedback.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for p in [input_path, raw_path, web_path]:
        try:
            os.unlink(p)
        except Exception:
            pass


