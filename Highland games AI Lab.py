import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess
import math

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────
import mediapipe as mp
mp_pose = mp.solutions.pose

# ── 13 key landmark IDs ───────────────────────────────────────────────────────
LM = {
    "nose":        0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}

KEY_LANDMARK_IDS = list(LM.values())

KEY_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

DOT_COLOR  = (0, 255, 120)
LINE_COLOR = (0, 180, 255)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Highland Games Throw Coach", layout="centered")

st.title("🏴 Highland Games Throw Coach")
st.write("Upload a throwing video to get a wireframe overlay and instant biomechanics coaching.")

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS EXPANDER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("⚙️ Settings", expanded=False):
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

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_video = st.file_uploader(
    "Upload Throw Video", type=["mp4", "mov", "avi", "mpeg4"]
)

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def angle_between(a, b, c) -> float:
    """Angle at joint B formed by points A-B-C, in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def pt(landmarks, name):
    lm = landmarks.landmark[LM[name]]
    return (lm.x, lm.y)


def vis(landmarks, name) -> float:
    return landmarks.landmark[LM[name]].visibility


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# BIOMECHANICS ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def analyse_landmarks(all_landmarks: list, event_name: str) -> dict:
    """
    Compute biomechanical metrics from a list of MediaPipe landmark objects.
    Returns a dict of scalar summary metrics used to generate coaching feedback.
    """
    if not all_landmarks:
        return {}

    l_elbow_angles   = []
    r_elbow_angles   = []
    l_knee_angles    = []
    r_knee_angles    = []
    spine_angles     = []
    hip_heights      = []
    wrist_heights    = []
    hip_shoulder_sep = []

    MIN_VIS = 0.20

    for lms in all_landmarks:
        def ok(*names):
            return all(vis(lms, n) > MIN_VIS for n in names)

        if ok("l_shoulder", "l_elbow", "l_wrist"):
            l_elbow_angles.append(angle_between(
                pt(lms, "l_shoulder"), pt(lms, "l_elbow"), pt(lms, "l_wrist")))

        if ok("r_shoulder", "r_elbow", "r_wrist"):
            r_elbow_angles.append(angle_between(
                pt(lms, "r_shoulder"), pt(lms, "r_elbow"), pt(lms, "r_wrist")))

        if ok("l_hip", "l_knee", "l_ankle"):
            l_knee_angles.append(angle_between(
                pt(lms, "l_hip"), pt(lms, "l_knee"), pt(lms, "l_ankle")))

        if ok("r_hip", "r_knee", "r_ankle"):
            r_knee_angles.append(angle_between(
                pt(lms, "r_hip"), pt(lms, "r_knee"), pt(lms, "r_ankle")))

        if ok("l_shoulder", "r_shoulder", "l_hip", "r_hip"):
            sh_mid = midpoint(pt(lms, "l_shoulder"), pt(lms, "r_shoulder"))
            hi_mid = midpoint(pt(lms, "l_hip"),      pt(lms, "r_hip"))
            dx = sh_mid[0] - hi_mid[0]
            dy = sh_mid[1] - hi_mid[1]
            spine_angles.append(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))
            hip_heights.append(hi_mid[1])

            # Hip-shoulder horizontal separation (as fraction of frame width)
            hip_shoulder_sep.append(abs(hi_mid[0] - sh_mid[0]))

        # Highest wrist (lowest normalised Y)
        if ok("l_wrist") and ok("r_wrist"):
            wrist_heights.append(min(pt(lms, "l_wrist")[1], pt(lms, "r_wrist")[1]))
        elif ok("l_wrist"):
            wrist_heights.append(pt(lms, "l_wrist")[1])
        elif ok("r_wrist"):
            wrist_heights.append(pt(lms, "r_wrist")[1])

    def smean(lst): return float(np.mean(lst))   if lst else None
    def smin(lst):  return float(np.min(lst))    if lst else None
    def smax(lst):  return float(np.max(lst))    if lst else None

    all_elbow = l_elbow_angles + r_elbow_angles
    all_knee  = l_knee_angles  + r_knee_angles

    return {
        "event":       event_name,
        "frames":      len(all_landmarks),
        "max_elbow":   smax(all_elbow),
        "mean_elbow":  smean(all_elbow),
        "min_knee":    smin(all_knee),
        "mean_knee":   smean(all_knee),
        "mean_spine":  smean(spine_angles),
        "mean_sep":    smean(hip_shoulder_sep),
        "min_wrist_y": smin(wrist_heights),
        "hip_range":   (smax(hip_heights) - smin(hip_heights)) if len(hip_heights) > 1 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COACHING REPORT
# ─────────────────────────────────────────────────────────────────────────────

def render_coaching_report(m: dict, event_label: str):
    st.subheader("🏆 Throw Coaching Report")
    st.caption(f"Event: **{event_label}** · Analysed across **{m['frames']}** sampled frames")

    # ── Scorecard ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Biomechanics Scorecard")

    col1, col2 = st.columns(2)

    def fmt(val, unit="°"):
        return f"{val:.1f}{unit}" if val is not None else "N/A"

    with col1:
        # Arm extension
        v = m["max_elbow"]
        if v is None:      icon = "⚪"
        elif v >= 155:     icon = "✅"
        elif v >= 130:     icon = "⚠️"
        else:              icon = "❌"
        st.markdown(f"{icon} **Arm extension (peak):** {fmt(v)}")

        # Knee bend
        v = m["min_knee"]
        if v is None:           icon = "⚪"
        elif 90 <= v <= 140:    icon = "✅"
        elif 70 <= v <= 155:    icon = "⚠️"
        else:                   icon = "❌"
        st.markdown(f"{icon} **Knee bend (power phase):** {fmt(v)}")

        # Spine
        v = m["mean_spine"]
        if v is None:    icon = "⚪"
        elif v <= 18:    icon = "✅"
        elif v <= 28:    icon = "⚠️"
        else:            icon = "❌"
        st.markdown(f"{icon} **Spine lean (avg):** {fmt(v)}")

    with col2:
        # Hip-shoulder separation
        v = m["mean_sep"]
        if v is None:           icon, disp = "⚪", "N/A"
        elif v * 100 >= 9:      icon, disp = "✅", f"{v*100:.1f}% of width"
        elif v * 100 >= 4:      icon, disp = "⚠️", f"{v*100:.1f}% — needs more rotation"
        else:                   icon, disp = "❌", f"{v*100:.1f}% — very limited"
        st.markdown(f"{icon} **Hip-shoulder separation:** {disp}")

        # Release height
        v = m["min_wrist_y"]
        if v is None:        icon, disp = "⚪", "N/A"
        elif v < 0.30:       icon, disp = "✅", "Above head ✓"
        elif v < 0.50:       icon, disp = "⚠️", "Mid-body — finish higher"
        else:                icon, disp = "❌", "Below hips — arm not following through"
        st.markdown(f"{icon} **Release / wrist height:** {disp}")

        # Hip drive
        v = m["hip_range"]
        if v is None:          icon, disp = "⚪", "N/A"
        elif v * 100 >= 8:     icon, disp = "✅", f"{v*100:.1f}% vertical range"
        elif v * 100 >= 3:     icon, disp = "⚠️", f"{v*100:.1f}% — push harder through legs"
        else:                  icon, disp = "❌", f"{v*100:.1f}% — very little leg drive"
        st.markdown(f"{icon} **Hip / leg drive:** {disp}")

    # ── Strengths ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💪 Strengths")
    strengths = []
    if m["max_elbow"] and m["max_elbow"] >= 155:
        strengths.append("**Full arm extension** — your throwing arm reaches near-straight at release, maximising lever length.")
    if m["min_knee"] and 90 <= m["min_knee"] <= 140:
        strengths.append("**Good leg loading** — knee angle shows you're sitting into the throw and using your legs.")
    if m["mean_spine"] and m["mean_spine"] <= 18:
        strengths.append("**Upright posture** — spine stays tall through the throw, keeping force directed forward.")
    if m["mean_sep"] and m["mean_sep"] * 100 >= 9:
        strengths.append("**Hip-shoulder separation** — hips are leading shoulders, creating the elastic stretch that generates power.")
    if m["min_wrist_y"] and m["min_wrist_y"] < 0.35:
        strengths.append("**High finish** — wrists reach above shoulder height, giving the implement a good launch angle.")
    if m["hip_range"] and m["hip_range"] * 100 >= 8:
        strengths.append("**Active leg drive** — measurable vertical hip movement shows you're pushing off the ground.")

    for s in (strengths or ["Keep working on the areas below — every improvement here adds distance."]):
        st.markdown(f"- {s}")

    # ── Top Improvements ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Top Improvements to Add Distance")

    improvements = []

    if m["max_elbow"] is not None and m["max_elbow"] < 155:
        improvements.append((
            "Fully extend your throwing arm at release",
            f"Your peak elbow angle is **{m['max_elbow']:.1f}°** — you want **165–180°**. "
            "A bent elbow shortens your effective lever by inches and costs significant distance. "
            "Think of your arm like a whip: it only works if it fully snaps out.\n\n"
            "**Drill — Wall Reach:** Stand arm's-length from a wall. "
            "Simulate your throw, extending until your knuckles just touch the wall at the finish. "
            "Repeat 20 reps focusing only on that final extension. "
            "Do this before every training session."
        ))

    if m["min_knee"] is not None and m["min_knee"] > 145:
        improvements.append((
            "Load your legs deeper in the power phase",
            f"Minimum knee angle is **{m['min_knee']:.1f}°** — elite throwers reach **100–130°** at peak load. "
            "Staying too upright means you're throwing with your upper body only, "
            "leaving your biggest muscles (quads, glutes) out of the movement.\n\n"
            "**Drill — Squat-to-Throw:** With a light medicine ball (3–5 kg), "
            "squat to 90° then explode upward, releasing the ball overhead at full extension. "
            "3 sets of 8 reps. Focus on the feeling of ground force transferring through your hips."
        ))
    elif m["min_knee"] is not None and m["min_knee"] < 80:
        improvements.append((
            "Avoid over-squatting — control your depth",
            f"Knee angle drops to **{m['min_knee']:.1f}°** which may cause balance loss and slow your transition. "
            "Aim for **100–130°** at peak load — deep enough to load, not so deep you get stuck."
        ))

    if m["mean_sep"] is not None and m["mean_sep"] * 100 < 8:
        improvements.append((
            "Drive your hips ahead of your shoulders (rotation separation)",
            f"Hip-shoulder separation is only **{m['mean_sep']*100:.1f}%** of frame width. "
            "Elite throwers show 10–15%+ because they initiate rotation from the hips, "
            "letting the shoulders lag behind. This stretch-shortening effect is the biggest "
            "single source of throwing power.\n\n"
            "**Drill — Banded Hip Fire:** Tie a resistance band around a post at hip height. "
            "Stand side-on to the post, band pulling your hip back. "
            "Practise firing your hip forward while keeping your throwing shoulder back. "
            "Feel the separation. 3 sets of 10 each side."
        ))

    if m["mean_spine"] is not None and m["mean_spine"] > 25:
        improvements.append((
            "Keep your spine taller through the throw",
            f"Average spine lean is **{m['mean_spine']:.1f}°** from vertical. "
            "Excessive forward lean redirects force downward and strains the lower back. "
            "Your chest should stay proud and tall through the power phase.\n\n"
            "**Cue:** Before every rep, take a deep breath and 'grow tall'. "
            "Imagine someone is pulling a string from the top of your head straight up. "
            "Hold that height as you throw."
        ))

    if m["min_wrist_y"] is not None and m["min_wrist_y"] > 0.50:
        improvements.append((
            "Finish the throw higher — drive through the release",
            "Wrists are staying at or below hip height, which means the implement is releasing "
            "at a very low angle. For maximum distance you need a 30–45° launch angle, "
            "which requires your arm to come all the way through and finish above your head.\n\n"
            "**Drill — Overhead Release Toss:** Using a light ball or sandbag, "
            "focus only on finishing with both hands above your forehead. "
            "Exaggerate the follow-through. Once it feels natural, add it back to your full throw."
        ))

    if m["hip_range"] is not None and m["hip_range"] * 100 < 4:
        improvements.append((
            "Drive harder off the ground — use your legs",
            "Very little vertical hip movement was detected. You may be throwing flat-footed. "
            "Rising onto the balls of your feet — or even leaving the ground — at release "
            "transfers your full body weight into the implement.\n\n"
            "**Cue:** 'Jump into the throw.' Your heels should be off the ground at the moment of release. "
            "If they're not, your legs are spectators, not contributors."
        ))

    if improvements:
        for i, (title, detail) in enumerate(improvements[:5], 1):
            with st.expander(f"**#{i} — {title}**", expanded=(i == 1)):
                st.markdown(detail)
    else:
        st.success("No major mechanical issues detected — your technique scores well across all "
                   "measured categories. Focus on increasing rotational speed and implement control.")

    # ── Distance Summary ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📏 Distance Potential Summary")

    if improvements:
        top   = improvements[0][0]
        rest  = [t for t, _ in improvements[1:3]]
        msg   = f"The biggest distance leak in this throw is **{top.lower()}**. "
        msg  += "Fix that first — it will have an immediate effect. "
        if rest:
            msg += f"After that, address **{rest[0].lower()}**"
            if len(rest) > 1:
                msg += f" and **{rest[1].lower()}**"
            msg += " for compounding gains."
    else:
        msg = ("Mechanics look solid across all measured categories. "
               "To keep adding distance: sharpen the hip-to-shoulder timing, "
               "increase rotational speed through plyometric training, "
               "and build single-leg stability for a stronger base.")
    st.info(msg)

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        "📥 Download Coaching Report",
        data=_build_text_report(m, event_label, improvements, msg),
        file_name="coaching_report.txt",
        mime="text/plain",
    )


def _build_text_report(m, event_label, improvements, summary) -> str:
    def fmt(v, unit="°"): return f"{v:.1f}{unit}" if v is not None else "N/A"
    sep = m["mean_sep"] * 100 if m["mean_sep"] else None
    hr  = m["hip_range"] * 100 if m["hip_range"] else None

    lines = [
        "HIGHLAND GAMES THROW COACHING REPORT",
        f"Event  : {event_label}",
        f"Frames : {m['frames']}",
        "",
        "── BIOMECHANICS SCORECARD ──────────────────────",
        f"Peak arm extension       : {fmt(m['max_elbow'])}",
        f"Knee bend (power phase)  : {fmt(m['min_knee'])}",
        f"Spine lean (avg)         : {fmt(m['mean_spine'])}",
        f"Hip-shoulder separation  : {fmt(sep, '%')}",
        f"Hip/leg drive range      : {fmt(hr, '%')}",
        "",
        "── TOP IMPROVEMENTS ────────────────────────────",
    ]
    for i, (title, detail) in enumerate(improvements[:5], 1):
        plain = detail.replace("**", "")
        lines.append(f"\n#{i}  {title}\n{plain}")

    lines += ["", "── SUMMARY ─────────────────────────────────────",
              summary.replace("**", "")]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO / POSE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def reencode_h264(src: str, dst: str) -> tuple:
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(candidate) or \
                subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
            ffmpeg = candidate
            break
    else:
        return False, "ffmpeg not found"
    r = subprocess.run(
        [ffmpeg, "-y", "-i", src,
         "-vcodec", "libx264", "-preset", "fast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
        capture_output=True, text=True, timeout=600,
    )
    ok = r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 500
    return (True, "") if ok else (False, r.stderr[-400:] if r.stderr else "empty output")


def pick_frame_indices(total: int, n: int) -> list:
    start = max(0, int(total * 0.05))
    end   = min(total - 1, int(total * 0.95))
    if start >= end:
        return list(range(min(n, total)))
    step = max(1, (end - start) // (n - 1))
    return [start + i * step for i in range(n)][:n]


def draw_13_landmarks(frame_bgr, landmarks, w, h):
    canvas     = frame_bgr.copy()
    diag       = (w ** 2 + h ** 2) ** 0.5
    dot_radius = max(2, int(diag * 0.005))
    line_thick = max(1, int(diag * 0.003))

    coords = {}
    for idx in KEY_LANDMARK_IDS:
        lm = landmarks.landmark[idx]
        if lm.visibility > 0.15:
            coords[idx] = (int(np.clip(lm.x, 0, 1) * w),
                           int(np.clip(lm.y, 0, 1) * h))

    for a, b in KEY_CONNECTIONS:
        if a in coords and b in coords:
            cv2.line(canvas, coords[a], coords[b], LINE_COLOR, line_thick, cv2.LINE_AA)

    for _, pt_coord in coords.items():
        cv2.circle(canvas, pt_coord, dot_radius + 1, (0, 0, 0), 1,  cv2.LINE_AA)
        cv2.circle(canvas, pt_coord, dot_radius,     DOT_COLOR, -1, cv2.LINE_AA)

    return canvas


def process_video(input_path: str, raw_out: str, sample_indices: set):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, [], 0, 0

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(720 / native_w, 404 / native_h)
    out_w = int(native_w * scale) // 2 * 2
    out_h = int(native_h * scale) // 2 * 2

    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    key_frames, sampled_lms = [], []
    frame_idx = detect_count = 0
    bar = st.progress(0, text="Analysing pose…")

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,  enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results  = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            disp     = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            annotated = disp.copy()

            if results.pose_landmarks:
                detect_count += 1
                annotated = draw_13_landmarks(disp, results.pose_landmarks, out_w, out_h)
                if frame_idx in sample_indices:
                    sampled_lms.append(results.pose_landmarks)
                    key_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            cv2.putText(annotated, f"{frame_idx}/{total}",
                        (8, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (160, 160, 160), 1)
            writer.write(annotated)

            frame_idx += 1
            if total > 0:
                bar.progress(min(frame_idx / total, 1.0),
                             text=f"Frame {frame_idx} / {total}")

    cap.release()
    writer.release()
    bar.empty()
    return key_frames, sampled_lms, frame_idx, detect_count


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_video is not None:

    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_video.read())
        input_path = tmp.name

    raw_path = input_path.replace(suffix, "_raw.mp4")
    web_path = input_path.replace(suffix, "_web.mp4")

    _cap   = cv2.VideoCapture(input_path)
    _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()

    # Sample 16 frames evenly across the throw
    sample_list = pick_frame_indices(_total, 16)
    sample_set  = set(sample_list)

    # ── Step 1: Process ───────────────────────────────────────────────────────
    key_frames, sampled_lms, total_frames, detect_count = process_video(
        input_path, raw_path, sample_set
    )

    if key_frames is None:
        st.error("Could not open video file. Please try a different file.")
        st.stop()

    detect_pct = int(detect_count / max(total_frames, 1) * 100)
    if detect_pct >= 60:
        st.success(f"✅  Pose detected in {detect_pct}% of {total_frames} frames.")
    elif detect_pct > 0:
        st.warning(f"⚠️  Pose only detected in {detect_pct}% of frames. "
                   "Try better lighting or ensure the full body is in frame.")
    else:
        st.error("❌  No pose detected. Ensure the athlete is fully visible in the frame.")

    # ── Step 2: Re-encode ─────────────────────────────────────────────────────
    with st.spinner("Preparing video for playback…"):
        ok, enc_err = reencode_h264(raw_path, web_path)

    if not ok:
        st.warning(f"Re-encode failed: `{enc_err}` — add `ffmpeg` to packages.txt")

    st.subheader("🎬 Wireframe Overlay")
    with open(web_path if ok else raw_path, "rb") as f:
        st.video(f.read())

    # ── Step 3: Key frame strip ───────────────────────────────────────────────
    if key_frames:
        st.subheader("🖼️ Sampled Key Frames")
        cols = st.columns(min(len(key_frames), 6))
        for col, (frame, idx) in zip(cols, zip(key_frames[:6], sample_list)):
            with col:
                st.image(frame, caption=f"Frame {idx}", use_container_width=True)

    # ── Step 4: Coaching report ───────────────────────────────────────────────
    if sampled_lms:
        event_label = event if event != "Auto-detect" else "Highland Games Throw"
        metrics = analyse_landmarks(sampled_lms, event_label)
        render_coaching_report(metrics, event_label)
    else:
        st.warning("Not enough pose data to generate a coaching report. "
                   "Ensure the athlete's full body is visible throughout the video.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for p in [input_path, raw_path, web_path]:
        try:
            os.unlink(p)
        except Exception:
            pass
