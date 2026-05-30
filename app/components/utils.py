"""
utils.py — Shared helpers for the Streamlit pages.
"""

import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ─── Image helpers ────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


def frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def load_image_safe(path: str) -> Image.Image | None:
    try:
        if Path(path).exists():
            return Image.open(path).convert("RGB")
    except Exception:
        pass
    return None


# ─── Streamlit UI helpers ─────────────────────────────────────────────────────

def page_config(title: str = "FaceVault", icon: str = "🎭"):
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_global_css():
    """Inject custom CSS for a polished dark theme."""
    st.markdown(
        """
        <style>
        /* ── Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(160deg, #0f0f1a 0%, #141428 100%);
            border-right: 1px solid #2a2a45;
        }

        /* ── Metric cards ── */
        [data-testid="metric-container"] {
            background: #1a1a2e;
            border: 1px solid #2a2a45;
            border-radius: 12px;
            padding: 1rem;
        }

        /* ── Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, #6c63ff, #4ecdc4);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        .stButton > button:hover { opacity: 0.85; }

        /* ── Success / Error ── */
        .stSuccess { border-left: 4px solid #4ecdc4; }
        .stError   { border-left: 4px solid #ff6b6b; }

        /* ── Code / mono ── */
        code { font-family: 'JetBrains Mono', monospace; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def confidence_badge(confidence: float) -> str:
    """Return an HTML badge coloured by confidence level."""
    if confidence >= 80:
        color, label = "#4ecdc4", "High"
    elif confidence >= 60:
        color, label = "#ffe66d", "Medium"
    else:
        color, label = "#ff6b6b", "Low"
    return (
        f'<span style="background:{color};color:#0f0f1a;'
        f'padding:2px 10px;border-radius:20px;font-weight:600;font-size:0.8rem;">'
        f'{label} {confidence:.0f}%</span>'
    )


def avatar_html(img: Image.Image, size: int = 80) -> str:
    """Render a face image as a circular HTML avatar."""
    b64 = image_to_base64(img.resize((size, size)))
    return (
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'style="width:{size}px;height:{size}px;border-radius:50%;'
        f'object-fit:cover;border:2px solid #6c63ff;" />'
    )