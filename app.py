import base64
import io
from datetime import datetime

import numpy as np
import requests
import streamlit as st
from PIL import Image

_LOCAL_URL = "http://localhost:8000"
_HF_URL    = "https://primeintern10-skinscope-backend.hf.space"

def _get_backend_url() -> str:
    try:
        requests.get(f"{_LOCAL_URL}/docs", timeout=2)
        return _LOCAL_URL
    except Exception:
        return _HF_URL

BACKEND_URL = _get_backend_url()

st.set_page_config(
    page_title="SkinScope — AI Skin Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Radio label fix ── */
[data-testid="stRadio"] > label:first-child { display: none !important; }

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #f0f0f0;
    margin-bottom: 2rem;
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-logo .logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.topbar-logo .logo-text {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f0f0f;
    letter-spacing: -0.3px;
}
.topbar-logo .logo-sub {
    font-size: 0.7rem;
    font-weight: 400;
    color: #888;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.topbar-tag {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6b7280;
    background: #f3f4f6;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
}

/* ── Section headings ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 12px;
}

/* ── Upload zone ── */
.upload-zone {
    border: 1.5px dashed #d1d5db;
    border-radius: 12px;
    padding: 24px;
    background: #fafafa;
    transition: border-color 0.2s;
}

/* ── Concern card ── */
.concern-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    background: #fff;
    border: 1px solid #f3f4f6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.concern-row:hover { border-color: #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.concern-left { display: flex; align-items: center; gap: 12px; flex: 1; }
.concern-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.concern-dot.high     { background: #ef4444; }
.concern-dot.moderate { background: #f59e0b; }
.concern-dot.low      { background: #10b981; }
.concern-name { font-size: 0.88rem; font-weight: 600; color: #111827; }
.concern-desc { font-size: 0.75rem; color: #9ca3af; margin-top: 1px; }
.concern-right { display: flex; align-items: center; gap: 12px; }
.concern-score {
    font-size: 0.88rem;
    font-weight: 700;
    min-width: 42px;
    text-align: right;
}
.concern-score.high     { color: #ef4444; }
.concern-score.moderate { color: #f59e0b; }
.concern-score.low      { color: #10b981; }
.concern-bar-wrap {
    width: 80px; height: 4px;
    background: #f3f4f6;
    border-radius: 4px;
    overflow: hidden;
}
.concern-bar {
    height: 4px;
    border-radius: 4px;
}
.concern-bar.high     { background: #ef4444; }
.concern-bar.moderate { background: #f59e0b; }
.concern-bar.low      { background: #10b981; }
.sev-pill {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.sev-pill.high     { background: #fef2f2; color: #dc2626; }
.sev-pill.moderate { background: #fffbeb; color: #d97706; }
.sev-pill.low      { background: #f0fdf4; color: #16a34a; }

/* ── Stat cards ── */
.stat-grid { display: flex; gap: 12px; margin-bottom: 20px; }
.stat-box {
    flex: 1;
    background: #fff;
    border: 1px solid #f3f4f6;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stat-box .s-label { font-size: 0.7rem; font-weight: 500; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.stat-box .s-value { font-size: 1.6rem; font-weight: 700; color: #111827; }
.stat-box .s-sub   { font-size: 0.72rem; color: #9ca3af; margin-top: 2px; }

/* ── Model tags ── */
.model-strip { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; }
.model-tag {
    font-size: 0.7rem; font-weight: 500;
    background: #f8fafc; color: #475569;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 3px 10px;
}

/* ── Feature table ── */
.feat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f9fafb;
    font-size: 0.82rem;
}
.feat-row:last-child { border-bottom: none; }
.feat-name { font-weight: 500; color: #374151; }
.feat-desc { font-size: 0.72rem; color: #9ca3af; margin-top: 1px; }
.feat-val  { font-weight: 600; color: #4f46e5; font-size: 0.85rem; }

/* ── Divider ── */
.thin-divider { border: none; border-top: 1px solid #f3f4f6; margin: 24px 0; }

/* ── Image caption ── */
.img-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase;
    color: #9ca3af; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key in ("uploaded_image", "detection_result", "analysis_result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Settings")
    detection_confidence = st.slider(
        "Detection confidence", 0.1, 1.0, 0.5, 0.05,
        help="Minimum confidence threshold for face detection",
    )

# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="logo-icon">🔬</div>
        <div>
            <div class="logo-text">SkinScope</div>
            <div class="logo-sub">AI Skin Analysis</div>
        </div>
    </div>
    <div class="topbar-tag">Powered by MediaPipe · ViT · EfficientNet</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_analysis, tab_debug = st.tabs(["Analysis", "Debug"])


# =============================================================================
# TAB 1 — ANALYSIS
# =============================================================================
with tab_analysis:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        input_method = st.radio(
            "input", ["Upload Image", "Take Photo"],
            horizontal=True, label_visibility="collapsed",
        )
        if input_method == "Upload Image":
            uploaded = st.file_uploader(
                "img", type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
            )
            if uploaded:
                st.session_state.uploaded_image = uploaded.getvalue()
                st.session_state.analysis_result = None
        else:
            camera = st.camera_input("cam", label_visibility="collapsed")
            if camera:
                st.session_state.uploaded_image = camera.getvalue()
                st.session_state.analysis_result = None

        if st.session_state.uploaded_image:
            st.markdown('<div class="img-label" style="margin-top:16px">Preview</div>', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, use_container_width=True)

    with col_right:
        if st.session_state.uploaded_image:
            analyze_btn = st.button(
                "Run Analysis", use_container_width=True,
                type="primary", key="analyze_default",
            )
            if analyze_btn:
                with st.spinner("Analysing skin…"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/analyze",
                            files={"file": ("image.png", st.session_state.uploaded_image, "image/png")},
                            data={"confidence": detection_confidence},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        st.session_state.analysis_result = resp.json()
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot reach backend at {BACKEND_URL}")
                    except requests.exceptions.HTTPError as e:
                        try:
                            detail = e.response.json().get("detail", str(e))
                        except Exception:
                            detail = e.response.text or str(e)
                        st.error(f"Analysis failed: {detail}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        if st.session_state.analysis_result:
            data     = st.session_state.analysis_result
            concerns = data["concerns"]

            high_c = sum(1 for c in concerns if c["severity"] == "High")
            mod_c  = sum(1 for c in concerns if c["severity"] == "Moderate")
            avg_s  = np.mean([c["score"] for c in concerns])

            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="s-label">Avg Score</div>
                    <div class="s-value">{avg_s:.0f}</div>
                    <div class="s-sub">out of 95</div>
                </div>
                <div class="stat-box">
                    <div class="s-label">High</div>
                    <div class="s-value" style="color:#ef4444">{high_c}</div>
                    <div class="s-sub">concerns</div>
                </div>
                <div class="stat-box">
                    <div class="s-label">Moderate</div>
                    <div class="s-value" style="color:#f59e0b">{mod_c}</div>
                    <div class="s-sub">concerns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Concern Breakdown</div>', unsafe_allow_html=True)
            for concern in concerns:
                sev   = concern["severity"].lower()
                score = concern["score"]
                pct   = int((score - 10) / 85 * 100)
                st.markdown(f"""
                <div class="concern-row">
                    <div class="concern-left">
                        <div class="concern-dot {sev}"></div>
                        <div>
                            <div class="concern-name">{concern['name']}</div>
                            <div class="concern-desc">{concern['description']}</div>
                        </div>
                    </div>
                    <div class="concern-right">
                        <div class="concern-bar-wrap">
                            <div class="concern-bar {sev}" style="width:{pct}%"></div>
                        </div>
                        <span class="concern-score {sev}">{score:.0f}</span>
                        <span class="sev-pill {sev}">{concern['severity']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif not st.session_state.uploaded_image:
            st.markdown("""
            <div style="color:#9ca3af; font-size:0.88rem; padding-top:40px; text-align:center;">
                Upload or capture an image to begin analysis.
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 2 — DEBUG
# =============================================================================
with tab_debug:
    col_l, col_r = st.columns([1.3, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        input_method_d = st.radio(
            "input_d", ["Upload Image", "Take Photo"],
            horizontal=True, label_visibility="collapsed",
        )
        if input_method_d == "Upload Image":
            uploaded_d = st.file_uploader(
                "img_d", type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed", key="uploader_debug",
            )
            if uploaded_d:
                st.session_state.uploaded_image = uploaded_d.getvalue()
                st.session_state.detection_result = None
                st.session_state.analysis_result  = None
        else:
            camera_d = st.camera_input("cam_d", label_visibility="collapsed", key="camera_debug")
            if camera_d:
                st.session_state.uploaded_image = camera_d.getvalue()
                st.session_state.detection_result = None
                st.session_state.analysis_result  = None

        if st.session_state.uploaded_image:
            st.markdown('<div class="img-label" style="margin-top:16px">Original</div>', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, use_container_width=True)

    with col_r:
        if st.session_state.uploaded_image:
            c1, c2 = st.columns(2)
            with c1:
                detect_btn = st.button("Detect Faces", use_container_width=True, key="detect_debug")
            with c2:
                analyze_btn_d = st.button("Full Analysis", use_container_width=True,
                                          type="primary", key="analyze_debug")

            if detect_btn:
                with st.spinner("Detecting…"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/detect",
                            files={"file": ("image.png", st.session_state.uploaded_image, "image/png")},
                            data={"confidence": detection_confidence},
                        )
                        resp.raise_for_status()
                        st.session_state.detection_result = resp.json()
                        st.session_state.analysis_result  = None
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot reach backend at {BACKEND_URL}")
                    except Exception as e:
                        st.error(f"Detection failed: {e}")

            if analyze_btn_d:
                with st.spinner("Running full analysis…"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/analyze",
                            files={"file": ("image.png", st.session_state.uploaded_image, "image/png")},
                            data={"confidence": detection_confidence},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        st.session_state.analysis_result  = resp.json()
                        st.session_state.detection_result = None
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot reach backend at {BACKEND_URL}")
                    except requests.exceptions.HTTPError as e:
                        try:
                            detail = e.response.json().get("detail", str(e))
                        except Exception:
                            detail = e.response.text or str(e)
                        st.error(f"Analysis failed: {detail}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

            result = st.session_state.analysis_result or st.session_state.detection_result
            if result and "annotated_image" in result:
                st.markdown('<div class="img-label" style="margin-top:16px">Annotated</div>', unsafe_allow_html=True)
                st.image(base64.b64decode(result["annotated_image"]), use_container_width=True)

            if st.session_state.analysis_result:
                crop_b64 = st.session_state.analysis_result.get("skin_crop_image")
                if crop_b64:
                    st.markdown('<div class="img-label" style="margin-top:16px">Skin Mask</div>', unsafe_allow_html=True)
                    st.image(base64.b64decode(crop_b64), use_container_width=True)

    # ── Debug results ──────────────────────────────────────────────────────────
    if st.session_state.analysis_result:
        data        = st.session_state.analysis_result
        concerns    = data["concerns"]
        features    = data["features"]
        models_used = data.get("models_used", [])

        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

        tags = "".join(f'<span class="model-tag">{m}</span>' for m in models_used)
        st.markdown(f'<div class="model-strip">{tags}</div>', unsafe_allow_html=True)

        high_c = sum(1 for c in concerns if c["severity"] == "High")
        mod_c  = sum(1 for c in concerns if c["severity"] == "Moderate")
        avg_s  = np.mean([c["score"] for c in concerns])

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="s-label">Avg Score</div>
                <div class="s-value">{avg_s:.0f}</div>
                <div class="s-sub">out of 95</div>
            </div>
            <div class="stat-box">
                <div class="s-label">High</div>
                <div class="s-value" style="color:#ef4444">{high_c}</div>
                <div class="s-sub">concerns</div>
            </div>
            <div class="stat-box">
                <div class="s-label">Moderate</div>
                <div class="s-value" style="color:#f59e0b">{mod_c}</div>
                <div class="s-sub">concerns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Concern Breakdown</div>', unsafe_allow_html=True)
        for concern in concerns:
            sev   = concern["severity"].lower()
            score = concern["score"]
            pct   = int((score - 10) / 85 * 100)
            st.markdown(f"""
            <div class="concern-row">
                <div class="concern-left">
                    <div class="concern-dot {sev}"></div>
                    <div>
                        <div class="concern-name">{concern['name']}</div>
                        <div class="concern-desc">{concern['description']}</div>
                    </div>
                </div>
                <div class="concern-right">
                    <div class="concern-bar-wrap">
                        <div class="concern-bar {sev}" style="width:{pct}%"></div>
                    </div>
                    <span class="concern-score {sev}">{score:.0f}</span>
                    <span class="sev-pill {sev}">{concern['severity']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Raw Feature Values"):
            feature_labels = {
                "redness":                ("Redness",           "HSV red-zone pixel ratio"),
                "oiliness":               ("Oiliness",          "Specular highlight ratio"),
                "brightness":             ("Brightness",        "Mean LAB L* normalised"),
                "texture_variance":       ("Texture Variance",  "Laplacian variance normalised"),
                "color_variance":         ("Colour Variance",   "RGB std-dev normalised"),
                "pigmentation":           ("Pigmentation",      "LAB b* std-dev normalised"),
                "dark_spot_ratio":        ("Dark Spot Ratio",   "Sub-median dark pixel fraction"),
                "local_redness_clusters": ("Redness Clusters",  "Relative inflamed blob count"),
                "saturation_inv":         ("Saturation Drop",   "Relative saturation CV"),
                "flakiness":              ("Flakiness",         "Local texture roughness"),
                "lab_uniformity":         ("LAB Uniformity",    "Inter-block brightness CV"),
            }
            rows = ""
            for key, (label, desc) in feature_labels.items():
                val = features.get(key, 0.0)
                pct = int(val * 100)
                rows += f"""
                <div class="feat-row">
                    <div>
                        <div class="feat-name">{label}</div>
                        <div class="feat-desc">{desc}</div>
                    </div>
                    <span class="feat-val">{val:.3f}</span>
                </div>"""
            st.markdown(f'<div style="padding:4px 0">{rows}</div>', unsafe_allow_html=True)

    elif st.session_state.detection_result:
        data = st.session_state.detection_result
        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Detection Results</div>', unsafe_allow_html=True)

        avg_conf = np.mean([f["confidence"] for f in data["face_details"]]) if data["face_count"] > 0 else 0
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-box">
                <div class="s-label">Faces Detected</div>
                <div class="s-value">{data['face_count']}</div>
            </div>
            <div class="stat-box">
                <div class="s-label">Avg Confidence</div>
                <div class="s-value">{avg_conf:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for face in data["face_details"]:
            st.markdown(f"""
            <div class="concern-row">
                <div class="concern-left">
                    <div class="concern-dot low"></div>
                    <div>
                        <div class="concern-name">Face #{face['face_id']}</div>
                        <div class="concern-desc">
                            Position ({face['x']}, {face['y']}) · Size {face['width']}×{face['height']}px
                        </div>
                    </div>
                </div>
                <span class="sev-pill low">{face['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#d1d5db; font-size:0.72rem; margin-top:48px; padding-top:16px; border-top:1px solid #f3f4f6;">
    SkinScope &nbsp;·&nbsp; MediaPipe &nbsp;·&nbsp; OpenCV &nbsp;·&nbsp; skintelligent-acne &nbsp;·&nbsp; EfficientNet-B0
</div>
""", unsafe_allow_html=True)
