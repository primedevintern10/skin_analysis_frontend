import base64
import io

import numpy as np
import requests
import streamlit as st
from PIL import Image

BACKEND_URL = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScope - AI Skin Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

.header { text-align: center; margin-bottom: 32px; }
.header h1 {
    font-size: 3em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: -1px;
}
.header p { color: #666; font-size: 1.1em; margin-top: 8px; }

.upload-section {
    background: linear-gradient(135deg,rgba(102,126,234,.06) 0%,rgba(118,75,162,.06) 100%);
    border: 2px solid rgba(102,126,234,.2);
    border-radius: 15px;
    padding: 28px;
    margin: 16px 0;
}

.concern-card {
    background: #fff;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
    border-left: 5px solid #667eea;
}
.concern-card.high   { border-left-color: #e74c3c; }
.concern-card.moderate { border-left-color: #f39c12; }
.concern-card.low    { border-left-color: #27ae60; }

.severity-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 700;
    margin-left: 8px;
}
.badge-high     { background:#fde8e8; color:#c0392b; }
.badge-moderate { background:#fef3cd; color:#856404; }
.badge-low      { background:#d4edda; color:#155724; }

.stat-card {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(102,126,234,.3);
}
.stat-card .label { font-size:.85em; opacity:.85; margin-bottom:6px; }
.stat-card .value { font-size:2em; font-weight:800; }

.model-badge {
    display: inline-block;
    background: rgba(102,126,234,.12);
    color: #5a67d8;
    border: 1px solid rgba(102,126,234,.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78em;
    font-weight: 600;
    margin: 3px 2px;
}

.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.88em;
}
.feature-row:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("uploaded_image", "detection_result", "analysis_result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>✨ SkinScope</h1>
    <p>AI-powered skin analysis · Face detection · 10-concern scoring</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📸 Input Source")
    input_method = st.radio(
        "Input method",
        ["Upload Image", "Take Photo"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    detection_confidence = st.slider(
        "Detection confidence",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Minimum confidence for face detection",
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "**SkinScope** runs a 3-layer pipeline:\n\n"
        "• MediaPipe face landmark masking\n"
        "• OpenCV photometric features\n"
        "• YOLOv11 acne detection\n"
        "• EfficientNet-B0 texture features\n\n"
        "YOLOv11 + EfficientNet run **in parallel threads** "
        "so total latency ≈ the slowest model, not their sum."
    )

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    if input_method == "Upload Image":
        st.markdown("### 📤 Upload Image")
        uploaded = st.file_uploader(
            "Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state.uploaded_image = uploaded.getvalue()
            st.session_state.detection_result = None
            st.session_state.analysis_result = None
    else:
        st.markdown("### 📷 Camera")
        camera = st.camera_input("Camera", label_visibility="collapsed")
        if camera:
            st.session_state.uploaded_image = camera.getvalue()
            st.session_state.detection_result = None
            st.session_state.analysis_result = None
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_image:
        st.markdown("### 🖼️ Original Image")
        st.image(st.session_state.uploaded_image, use_container_width=True)

with right:
    if st.session_state.uploaded_image:
        st.markdown("### 🚀 Actions")
        col_a, col_b = st.columns(2)

        with col_a:
            detect_btn = st.button("🔍 Detect Faces", use_container_width=True)
        with col_b:
            analyze_btn = st.button("🧬 Full Analysis", use_container_width=True, type="primary")

        # ── Quick face detection ──────────────────────────────────────────
        if detect_btn:
            with st.spinner("Detecting faces …"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/detect",
                        files={"file": ("image.png", st.session_state.uploaded_image, "image/png")},
                        data={"confidence": detection_confidence},
                    )
                    resp.raise_for_status()
                    st.session_state.detection_result = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot reach backend at {BACKEND_URL}")
                except Exception as e:
                    st.error(f"Detection failed: {e}")

        # ── Full analysis ─────────────────────────────────────────────────
        if analyze_btn:
            with st.spinner("Running full skin analysis … (first run downloads models)"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/analyze",
                        files={"file": ("image.png", st.session_state.uploaded_image, "image/png")},
                        data={"confidence": detection_confidence},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    st.session_state.analysis_result = resp.json()
                    st.session_state.detection_result = None   # analysis supersedes it
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot reach backend at {BACKEND_URL}")
                except requests.exceptions.HTTPError as e:
                    detail = e.response.json().get("detail", str(e))
                    st.error(f"Analysis failed: {detail}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        # ── Show annotated image (detect OR analyse) ──────────────────────
        result = st.session_state.analysis_result or st.session_state.detection_result
        if result and "annotated_image" in result:
            annotated = base64.b64decode(result["annotated_image"])
            st.markdown("### 📸 Annotated Image")
            st.image(annotated, use_container_width=True)

        # ── Skin crop preview ─────────────────────────────────────────────
        if st.session_state.analysis_result and "skin_crop_image" in st.session_state.analysis_result:
            crop_bytes = base64.b64decode(st.session_state.analysis_result["skin_crop_image"])
            st.markdown("### 🎭 Skin Mask")
            st.image(crop_bytes, use_container_width=True,
                     caption="Only skin pixels used for analysis")

# ── Analysis results section ──────────────────────────────────────────────────
if st.session_state.analysis_result:
    data = st.session_state.analysis_result
    concerns = data["concerns"]
    features = data["features"]
    models_used = data.get("models_used", [])

    st.markdown("---")

    # Models used badges
    badges = "".join(f'<span class="model-badge">⚡ {m}</span>' for m in models_used)
    st.markdown(f"**Models used:** {badges}", unsafe_allow_html=True)

    st.markdown("### 📊 Skin Analysis Results")

    # Top-level stats
    high_count = sum(1 for c in concerns if c["severity"] == "High")
    mod_count  = sum(1 for c in concerns if c["severity"] == "Moderate")
    avg_score  = np.mean([c["score"] for c in concerns])

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Avg Concern Score</div>
            <div class="value">{avg_score:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">High Concerns</div>
            <div class="value">{high_count}</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Moderate Concerns</div>
            <div class="value">{mod_count}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Concern cards + progress bars
    st.markdown("#### Concern Breakdown")
    for concern in concerns:
        sev   = concern["severity"].lower()
        score = concern["score"]
        color = {"high": "#e74c3c", "moderate": "#f39c12", "low": "#27ae60"}[sev]
        badge_cls = f"badge-{sev}"

        st.markdown(f"""
        <div class="concern-card {sev}">
            <strong>{concern['name']}</strong>
            <span class="severity-badge {badge_cls}">{concern['severity']}</span>
            <span style="float:right;font-weight:700;color:{color}">{score:.0f}/95</span>
            <div style="font-size:.82em;color:#888;margin-top:3px">{concern['description']}</div>
        </div>""", unsafe_allow_html=True)

        st.progress(int((score - 10) / 85 * 100))

    # Raw feature values (collapsible)
    with st.expander("🔬 Raw OpenCV Feature Values"):
        feature_labels = {
            "redness":          ("Redness",          "HSV red-zone pixel ratio"),
            "oiliness":         ("Oiliness / Shine",  "Specular highlight ratio"),
            "brightness":       ("Brightness",        "Mean LAB L* normalised"),
            "texture_variance": ("Texture Variance",  "Laplacian variance normalised"),
            "color_variance":   ("Colour Variance",   "RGB std-dev normalised"),
            "pigmentation":     ("Pigmentation",      "LAB b* std-dev normalised"),
            "dark_spot_ratio":  ("Dark Spot Ratio",   "Sub-median dark pixel fraction"),
        }
        rows = ""
        for key, (label, desc) in feature_labels.items():
            val = features.get(key, 0.0)
            rows += f"""
            <div class="feature-row">
                <span><strong>{label}</strong> <span style="color:#999">— {desc}</span></span>
                <span style="font-weight:700;color:#5a67d8">{val:.3f}</span>
            </div>"""
        st.markdown(f'<div style="padding:4px 0">{rows}</div>', unsafe_allow_html=True)

# ── Detection-only results (no full analysis) ─────────────────────────────────
elif st.session_state.detection_result:
    data = st.session_state.detection_result
    st.markdown("---")
    st.markdown("### 📊 Detection Results")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Faces Detected</div>
            <div class="value">{data['face_count']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        if data["face_count"] > 0:
            avg_conf = np.mean([f["confidence"] for f in data["face_details"]])
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">Avg Confidence</div>
                <div class="value">{avg_conf:.1%}</div>
            </div>""", unsafe_allow_html=True)

    for face in data["face_details"]:
        st.markdown(f"""
        <div class="concern-card low">
            <strong>Face #{face['face_id']}</strong> —
            Confidence: {face['confidence']:.2%} |
            Position: ({face['x']}, {face['y']}) |
            Size: {face['width']}×{face['height']}px
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#aaa;font-size:.85em;margin-top:32px">
    🔬 MediaPipe · OpenCV · YOLOv11 · EfficientNet-B0 · Built with Streamlit
</div>
""", unsafe_allow_html=True)
