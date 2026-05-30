"""
main.py — FaceVault Home Dashboard
Entry point for the Streamlit multi-page app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Bootstrap ──────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from components.database import init_db, get_all_persons, get_today_summary, get_attendance_logs
from components.face_engine import cache
from components.utils import page_config, inject_global_css

page_config("FaceVault — Home", "🎭")
inject_global_css()
init_db()

# ── Reload encoding cache ──────────────────────────────────────────────────────
if "cache_loaded" not in st.session_state:
    persons = get_all_persons()
    cache.load_from_db(persons)
    st.session_state["cache_loaded"] = True


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 1rem;">
        <h1 style="font-size:3rem;font-weight:700;
                   background:linear-gradient(135deg,#6c63ff,#4ecdc4);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🎭 FaceVault
        </h1>
        <p style="color:#888;font-size:1.1rem;margin-top:-0.5rem;">
            Intelligent Face Recognition & Attendance System
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
summary  = get_today_summary()
persons  = get_all_persons()
all_logs = get_attendance_logs()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Registered Faces", len(persons))
col2.metric("✅ Present Today",     summary["present"])
col3.metric("❓ Unknown Today",     summary["unknown"])
col4.metric("📋 Total Logs",        len(all_logs))

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
left, right = st.columns(2)

# -- Attendance trend (last 14 days) --
with left:
    st.subheader("📈 Attendance – Last 14 Days")
    logs_df = pd.DataFrame(all_logs) if all_logs else pd.DataFrame()

    if not logs_df.empty:
        logs_df["log_date"] = pd.to_datetime(logs_df["log_date"])
        cutoff = pd.Timestamp(date.today() - timedelta(days=14))
        trend  = (
            logs_df[logs_df["log_date"] >= cutoff]
            .groupby("log_date")
            .size()
            .reset_index(name="count")
        )
        fig = px.area(
            trend, x="log_date", y="count",
            color_discrete_sequence=["#6c63ff"],
            labels={"log_date": "Date", "count": "Recognitions"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attendance data yet. Start recognising faces!")

# -- Department breakdown --
with right:
    st.subheader("🏢 Registered by Department")
    if persons:
        depts = pd.DataFrame(persons)["department"].fillna("Unassigned").value_counts()
        fig2  = px.pie(
            values=depts.values, names=depts.index,
            color_discrete_sequence=px.colors.sequential.Plasma,
            hole=0.4,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No persons registered yet.")

st.divider()

# ── Quick nav cards ────────────────────────────────────────────────────────────
st.subheader("🚀 Quick Actions")
q1, q2, q3, q4 = st.columns(4)

def nav_card(col, icon, title, desc, page):
    col.markdown(
        f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a45;border-radius:12px;
                    padding:1.2rem;text-align:center;height:140px;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:600;margin:.4rem 0;">{title}</div>
            <div style="color:#888;font-size:.85rem;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

nav_card(q1, "📸", "Register",  "Enroll a new face",          "pages/1_Register_Face.py")
nav_card(q2, "🔍", "Recognize", "Live webcam recognition",    "pages/2_Recognize_Face.py")
nav_card(q3, "📋", "Attendance","View & export logs",         "pages/3_Attendance_Log.py")
nav_card(q4, "🗄️", "Database",  "Manage enrolled faces",      "pages/4_Manage_Database.py")

st.caption(f"FaceVault v1.0 · Today: {date.today().strftime('%A, %d %B %Y')}")