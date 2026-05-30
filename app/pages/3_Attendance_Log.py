"""
3_Attendance_Log.py — View, filter, and export attendance records.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

from components.database import init_db, get_attendance_logs, get_today_summary
from components.utils import page_config, inject_global_css, confidence_badge

page_config("Attendance Log", "📋")
inject_global_css()
init_db()

st.title("📋 Attendance Log")
st.caption("Filter, analyse, and export attendance records.")
st.divider()

# ── Today KPIs ────────────────────────────────────────────────────────────────
summary = get_today_summary()
k1, k2, k3 = st.columns(3)
k1.metric("✅ Present Today",  summary["present"])
k2.metric("❓ Unknown Today",  summary["unknown"])
k3.metric("📅 Date",           summary["date"])

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("🔎 Filters", expanded=True):
    fc1, fc2, fc3 = st.columns(3)
    start_date = fc1.date_input("From", value=date.today() - timedelta(days=7))
    end_date   = fc2.date_input("To",   value=date.today())
    name_filter = fc3.text_input("Name contains", placeholder="Leave blank for all")

logs = get_attendance_logs(
    start_date=start_date,
    end_date=end_date,
    person_name=name_filter.strip() or None,
)

df = pd.DataFrame(logs) if logs else pd.DataFrame()

if df.empty:
    st.info("No records match the current filters.")
    st.stop()

# ── Table ─────────────────────────────────────────────────────────────────────
st.subheader(f"📊 {len(df)} Record(s) Found")

display_df = df[["person_name", "status", "confidence", "log_date", "log_time"]].copy()
display_df.columns = ["Name", "Status", "Confidence %", "Date", "Time"]
display_df["Time"] = pd.to_datetime(display_df["Time"]).dt.strftime("%H:%M:%S")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Confidence %": st.column_config.ProgressColumn(
            "Confidence %", min_value=0, max_value=100, format="%.0f%%"
        ),
        "Status": st.column_config.TextColumn("Status"),
    },
)

# ── Charts ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Daily Attendance")
    df["log_date"] = pd.to_datetime(df["log_date"])
    daily = df.groupby("log_date").size().reset_index(name="count")
    fig = px.bar(
        daily, x="log_date", y="count",
        color_discrete_sequence=["#6c63ff"],
        labels={"log_date": "Date", "count": "Count"},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("👥 Top Attendees")
    top = df[df["status"] == "Present"]["person_name"].value_counts().head(10)
    if not top.empty:
        fig2 = px.bar(
            x=top.values, y=top.index, orientation="h",
            color_discrete_sequence=["#4ecdc4"],
            labels={"x": "Count", "y": "Name"},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ccc")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("📤 Export")
ex1, ex2 = st.columns(2)

csv_bytes = display_df.to_csv(index=False).encode("utf-8")
ex1.download_button(
    "⬇️ Download CSV",
    data=csv_bytes,
    file_name=f"attendance_{start_date}_to_{end_date}.csv",
    mime="text/csv",
    use_container_width=True,
)

try:
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        display_df.to_excel(writer, index=False)
    ex2.download_button(
        "⬇️ Download Excel",
        data=buffer.getvalue(),
        file_name=f"attendance_{start_date}_to_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except ImportError:
    ex2.info("Install `openpyxl` for Excel export.")