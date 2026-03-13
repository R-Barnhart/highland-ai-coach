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
# MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────
import mediapipe as mp
mp_pose = mp.solutions.pose

# ── 13-landmark skeleton ─────────────────────────────────────────────────────
# Exactly 13 joints covering the full body for throw analysis
KEY_LANDMARK_IDS = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
]

KEY_CONNECTIONS = [
    (11, 12),  # shoulder bar
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),            # hip bar
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

DOT_COLOR  = (0, 255, 120)   # bright green joints
LINE_COLOR = (0, 180, 255)   # sky-blue bones
# Sizes are computed dynamically per frame — see draw_13_landmarks()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Highland Games AI Coach", layout="centered")

st.title("🏴 Highland Games AI Throw Coach")
st.write("Upload a throwing video to get a wireframe overlay and Gemini AI coaching feedback.")

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS — collapsed expander (click ⚙️ Settings to open)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("⚙️ Settings", expanded=False):
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get yours free at aistudio.google.com/app/apikey",
    )
    if gemini_key:
        st.success("API key saved ✓", icon="🔐")

    col_a, col_b = st.columns(2)
    with col_a:
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
    with col_b:
        num_frames = st.slider(
            "Key frames sent to Gemini", 2, 6, 4,
            help="Fewer frames = lower quota usage.",
        )

    st.caption("💡 Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com/app/apikey). "
               "Enable billing for higher rate limits.")

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_video = st.file_uploader(
    "Upload Throw Video", type=["mp4", "mov", "avi", "mpeg4"]
)

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


def draw_13_landmarks(frame_bgr: np.ndarray, landmarks, w: int, h: int) -> np.ndarray:
    """Draw only the 13 key joints and their connections onto frame_bgr.

    Dot radius and line thickness scale with frame diagonal so markers
    look proportional on any resolution — not too big, not too small.
    """
    canvas = frame_bgr.copy()

    # Scale sizes to frame diagonal — kept deliberately small so dots
    # sit precisely on the joint rather than swamping it
    diag       = (w ** 2 + h ** 2) ** 0.5
    dot_radius = max(2, int(diag * 0.005))   # ~3-4 px on 720p
    line_thick = max(1, int(diag * 0.003))   # ~2 px on 720p

    # Build pixel-coord dict — only include joints MediaPipe is confident about
    coords = {}
    for idx in KEY_LANDMARK_IDS:
        lm = landmarks.landmark[idx]
        if lm.visibility > 0.15:   # low threshold → keep more joints visible
            x = int(np.clip(lm.x, 0.0, 1.0) * w)
            y = int(np.clip(lm.y, 0.0, 1.0) * h)
            coords[idx] = (x, y)

    # Draw bones first (behind dots)
    for a, b in KEY_CONNECTIONS:
        if a in coords and b in coords:
            cv2.line(canvas, coords[a], coords[b], LINE_COLOR, line_thick, cv2.LINE_AA)

    # Draw joint dots on top — thin black ring for contrast, filled green centre
    for idx, pt in coords.items():
        cv2.circle(canvas, pt, dot_radius + 1, (0, 0, 0), 1,  cv2.LINE_AA)  # outline
        cv2.circle(canvas, pt, dot_radius,     DOT_COLOR, -1, cv2.LINE_AA)  # fill

    return canvas


def gemini_coaching(api_key: str, frames_rgb: list, event_name: str) -> str:
    """Send annotated key frames to Gemini and return coaching text."""
    import google.generativeai as genai

    genai.configure(api_key=api_key, transport="rest")  # REST avoids gRPC header errors on Streamlit Cloud
    model = genai.GenerativeModel("gemini-2.0-flash")

    system_prompt = f"""You are an elite Highland Games coach and sports biomechanics expert
with 20+ years of competition and coaching experience across all Highland Games throwing events.

You are analysing {len(frames_rgb)} sequential key frames from a throw video.
Each frame shows a 13-point wireframe overlay (green dots = joints, blue lines = skeleton).
The 13 tracked joints are: nose, both shoulders, elbows, wrists, hips, knees, and ankles.

Your job is to give the thrower SPECIFIC, ACTIONABLE feedback that will directly add distance
to their throw. Study the skeleton positions carefully across all frames before writing.

Structure your response exactly as follows:

---

## 🎯 Event Identified
State the event name and what visual cues confirmed it.

---

## 📽️ Phase-by-Phase Breakdown
Walk through each throw phase visible in the frames:
- **Setup / Stance** — foot width, weight distribution, grip position
- **Wind-up / Approach** — rotation speed, counter-movement, implement path
- **Power Phase** — hip drive, leg push, trunk rotation sequence
- **Release Point** — arm extension, release angle, hand position at release
- **Follow-through** — balance, recovery, foul-line control

For each phase, reference what the skeleton shows (e.g. "the left hip landmark leads the shoulder by X frames, indicating good separation").

---

## ⚙️ Biomechanics Analysis
Analyse these specific factors and score each as ✅ Good / ⚠️ Needs Work / ❌ Issue:
- **Hip-Shoulder Separation** — is the hip rotating before the shoulder?
- **Foot Placement & Base** — width, toe angle, heel lift timing
- **Arm Path & Extension** — is the throwing arm fully extended at release?
- **Spine Angle** — upright vs. optimal lean for the event
- **Head & Eye Line** — is the head staying neutral or pulling early?
- **Release Angle** — is the implement leaving at the optimal angle for distance?

---

## 💪 What You're Doing Well
List 2-3 genuine strengths visible in the skeleton. Be specific — reference actual joint positions or timing you can see in the frames.

---

## 🚀 Top 3 Changes to Add Distance RIGHT NOW
These must be the highest-ROI fixes — the changes that will add the most distance fastest:

1. **[Fix name]** — What is wrong, exactly what to change, and why it will add distance.
2. **[Fix name]** — What is wrong, exactly what to change, and why it will add distance.
3. **[Fix name]** — What is wrong, exactly what to change, and why it will add distance.

---

## 🏋️ Recommended Drills
Provide 2-3 targeted drills that directly address the weaknesses you identified:
- **Drill name** — How to perform it, sets/reps, and which fault it corrects.

---

## 📏 Distance Potential Summary
Give a plain-English summary: "Based on what I see, the biggest distance leak is X. Fix that first and you should see immediate improvement. Secondary gains will come from Y and Z."

---

Be direct, technically precise, and encouraging. Assume the athlete is serious about improving."""

    # Build content parts: text prompt then alternating label + image
    parts = [system_prompt]
    for i, frame in enumerate(frames_rgb):
        parts.append(f"\nFrame {i + 1} of {len(frames_rgb)}:")
        pil_img = Image.fromarray(frame)
        parts.append(pil_img)

    response = model.generate_content(parts)
    return response.text


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_W = 720
DISPLAY_H = 404  # 16:9


def process_video(input_path: str, raw_out: str, key_indices: set):
    """
    Draw 13-point pose wireframe on every frame, write to raw_out.
    Collect key frames (RGB) at key_indices for Gemini.
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
    scale = min(DISPLAY_W / native_w, DISPLAY_H / native_h)
    out_w = int(native_w * scale) // 2 * 2   # keep even
    out_h = int(native_h * scale) // 2 * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (out_w, out_h))

    key_frames         = []
    frame_index        = 0
    landmarks_detected = 0
    bar = st.progress(0, text="Processing pose wireframe…")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # bundled lite model — no download needed on Streamlit Cloud
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Run pose on the NATIVE resolution for best accuracy ──────────
            rgb_native = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results    = pose.process(rgb_native)

            # ── Resize frame to display size AFTER detection ─────────────────
            frame_display = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            annotated     = frame_display.copy()

            if results.pose_landmarks:
                landmarks_detected += 1
                # Draw on the display-size frame; landmarks use normalised [0,1]
                # coords so they map correctly to any output resolution
                annotated = draw_13_landmarks(
                    frame_display, results.pose_landmarks, out_w, out_h
                )

            # Subtle frame counter
            cv2.putText(annotated, f"{frame_index}/{total}",
                        (8, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 160), 1)

            writer.write(annotated)

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

    if not gemini_key:
        st.warning("⚠️  Open **⚙️ Settings** above and enter your Gemini API key to enable AI coaching.")

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
        st.subheader("🖼️ Key Frames Sent to Gemini")
        cols = st.columns(len(key_frames))
        for col, (frame, idx) in zip(cols, zip(key_frames, key_idx_list)):
            with col:
                st.image(frame, caption=f"Frame {idx}", use_container_width=True)

    # ── Step 4: Gemini coaching ───────────────────────────────────────────────
    if gemini_key and key_frames:
        st.subheader("🏆 Gemini 2.0 Flash Coaching Feedback")
        event_label = event if event != "Auto-detect" else "Highland Games throw"

        with st.spinner("Gemini is analysing your throw…"):
            try:
                feedback = gemini_coaching(gemini_key, key_frames, event_label)
                st.markdown(feedback)

                st.download_button(
                    "📥 Download Coaching Report",
                    data=feedback,
                    file_name="coaching_report.txt",
                    mime="text/plain",
                )
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    import re
                    wait = re.search(r'retry.*?(\d+)s', err, re.IGNORECASE)
                    wait_msg = f" Please wait **{wait.group(1)} seconds** then try again." if wait else ""
                    st.error(
                        "⏱️ **Gemini free-tier quota exceeded.**" + wait_msg + "\n\n"
                        "**Quick fixes:**\n"
                        "- Reduce the *Key frames* slider to **2**\n"
                        "- Wait a minute and retry\n"
                        "- [Enable billing](https://ai.google.dev/gemini-api/docs/rate-limits) "
                        "on your Google AI account for higher limits"
                    )
                else:
                    st.error(f"Gemini error: {e}")

    elif not gemini_key:
        st.info("Open **⚙️ Settings** above and add your Gemini API key to get AI coaching feedback.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for p in [input_path, raw_path, web_path]:
        try:
            os.unlink(p)
        except Exception:
            pass
