"""
4_Manage_Database.py — View, search, and remove enrolled faces.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd

from components.database import init_db, get_all_persons, delete_person
from components.face_engine import cache
from components.utils import page_config, inject_global_css, load_image_safe, avatar_html

page_config("Manage Database", "🗄️")
inject_global_css()
init_db()

st.title("🗄️ Face Database")
st.caption("Browse and manage all enrolled persons.")
st.divider()

persons = get_all_persons()

if not persons:
    st.info("No persons registered yet. Go to **Register Face** to get started.")
    st.stop()

# ── Search bar ────────────────────────────────────────────────────────────────
query = st.text_input("🔍 Search by name or department", placeholder="Type to filter…")
filtered = [
    p for p in persons
    if query.lower() in p["name"].lower()
    or query.lower() in (p["department"] or "").lower()
] if query else persons

st.markdown(f"**{len(filtered)}** person(s) found")
st.divider()

# ── Grid of person cards ──────────────────────────────────────────────────────
COLS = 3
rows = [filtered[i:i+COLS] for i in range(0, len(filtered), COLS)]

for row in rows:
    cols = st.columns(COLS)
    for col, person in zip(cols, row):
        with col:
            face_img = load_image_safe(person["face_image"])

            with st.container():
                if face_img:
                    st.markdown(avatar_html(face_img, size=90), unsafe_allow_html=True)
                else:
                    st.markdown("🧑", unsafe_allow_html=True)

                st.markdown(
                    f"**{person['name']}**  \n"
                    f"🏢 {person['department'] or 'N/A'}  \n"
                    f"🆔 {person['employee_id'] or 'N/A'}  \n"
                    f"📧 {person['email'] or 'N/A'}  \n"
                    f"<span style='color:#888;font-size:.75rem;'>"
                    f"Registered: {person['registered_at'][:10]}</span>",
                    unsafe_allow_html=True,
                )

                if st.button(
                    "🗑️ Remove", key=f"del_{person['id']}",
                    help=f"Remove {person['name']} from the system",
                ):
                    st.session_state[f"confirm_{person['id']}"] = True

                if st.session_state.get(f"confirm_{person['id']}"):
                    st.warning(f"Remove **{person['name']}**?")
                    yes, no = st.columns(2)
                    if yes.button("Yes, remove", key=f"yes_{person['id']}"):
                        delete_person(person["id"])
                        cache.remove(person["id"])
                        st.success(f"✅ {person['name']} removed.")
                        del st.session_state[f"confirm_{person['id']}"]
                        st.rerun()
                    if no.button("Cancel", key=f"no_{person['id']}"):
                        del st.session_state[f"confirm_{person['id']}"]
                        st.rerun()

            st.markdown("---")

# ── Export list ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("📤 Export Person List")
df = pd.DataFrame(persons)[["id", "name", "employee_id", "department", "email", "registered_at"]]
df.columns = ["ID", "Name", "Employee ID", "Department", "Email", "Registered At"]

st.download_button(
    "⬇️ Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="registered_persons.csv",
    mime="text/csv",
    use_container_width=True,
)