# app.py - Navigation Hub
import streamlit as st
import pandas as pd
from PIL import Image
import os
import json
import datetime

import os
print(f"Current working directory: {os.getcwd()}")
print(f"Pages folder exists: {os.path.exists('pages')}")
print(f"pole.py exists: {os.path.exists('pages/pole.py')}")

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "gis_records.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
st.set_page_config(page_title="PolePad AI", layout="wide")

# ---------------------------
# Utility functions
# ---------------------------
def load_gis(path=CSV_PATH):
    df = pd.read_csv(path)
    return df

def get_flagged_poles(out_dir=REPORTS_DIR):
    """Enumerate flagged poles from reports"""
    poles = []
    if os.path.isdir(out_dir):
        for fname in os.listdir(out_dir):
            if fname.lower().endswith(".json"):
                try:
                    with open(os.path.join(out_dir, fname)) as f:
                        data = json.load(f)
                    if data.get("mismatches"):
                        poles.append(data.get("pole_id"))
                except Exception:
                    pass
    return sorted(set(poles))

def navigate_to_pole(pole_id):
    """Helper: navigate to pole.py with pole_id via session state"""
    st.session_state.selected_pole = pole_id
    st.switch_page("pages/pole.py")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("PolePad AI — Utility inspection made easy")
st.subheader("Defects flagged by AI, Reviewed by You")

# Load GIS
try:
    gis_df = load_gis(CSV_PATH)
except Exception as e:
    st.error(f"Could not load GIS CSV at {CSV_PATH}: {e}")
    st.stop()

col1, col2 = st.columns(2)

# ===== SECTION 1: UPLOAD IMAGE OR SELECT POLE =====
with col1:
    st.header("New Inspection")
    st.markdown("Upload a pole image or select a pole ID → review details & compare with GIS.")
    
    uploaded = st.file_uploader("Upload an inspection image (optional)", type=["jpg", "jpeg", "png"])
    
    st.write("**Or select a pole to review:**")
    pole_list = list(gis_df["pole_id"].values)
    selected_pole = st.selectbox("Select pole_id", options=[""] + pole_list, index=0, key="section1")
    
    if selected_pole:
        if st.button("Review this pole", key="btn_section1"):
            navigate_to_pole(selected_pole)
    
    if uploaded:
        st.image(Image.open(uploaded).convert("RGB"), caption="Uploaded image", use_column_width=True)
        if st.button("Go to review", key="btn_upload"):
            navigate_to_pole(selected_pole)
    else:
        st.info("Select a pole from the left section first.")

# ===== SECTION 2: REVIEW FLAGGED POLES =====
with col2:
    st.header("Flagged Poles")
    st.markdown("Review poles that have mismatches & need attention.")
    
    flagged = get_flagged_poles()
    if flagged:
        st.write(f"**{len(flagged)} pole(s) flagged**")
        selected_flagged = st.selectbox("Select flagged pole", options=[""] + flagged, key="section2")
    
        if selected_flagged:
            if st.button("Review flagged pole", key="btn_section2"):
                navigate_to_pole(selected_flagged)
    else:
        st.info("No flagged reports found.")



st.sidebar.markdown("## Notes")
st.sidebar.write("- GIS records must be present as `gis_records.csv` in the project folder.")
st.sidebar.write("- AI/GIS comparison happens on the Pole Review page.")
st.sidebar.write("- Reports are saved to the `reports/` folder.")
