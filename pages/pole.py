import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import pytesseract
import io
import os
import json
import datetime

CSV_PATH = "gis_records.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
st.set_page_config(page_title="Pole Review", layout="wide")

# ---------------------------
# Utility functions (reuse your compare logic)
# ---------------------------
def load_gis(path=CSV_PATH):
    df = pd.read_csv(path)
    return df

def get_pole_record(df, pole_id):
    rec = df[df["pole_id"] == pole_id]
    if rec.empty:
        return None
    return rec.iloc[0].to_dict()

def compare(ai, gis):
    mismatches = []
    # Compare vegetation
    if str(ai["vegetation"]).strip().lower() != str(gis["expected_vegetation"]).strip().lower():
        mismatches.append(f"Vegetation mismatch: GIS={gis['expected_vegetation']} vs AI={ai['vegetation']}")
    # Compare guy guard
    if str(ai["guy_guard"]).strip().lower() != str(gis["expected_guy_guard"]).strip().lower():
        mismatches.append(f"Guy guard mismatch: GIS={gis['expected_guy_guard']} vs AI={ai['guy_guard']}")
    # Compare pole type
    if str(ai["pole_type"]).strip().lower() != str(gis['pole_type']).strip().lower():
        mismatches.append(f"Pole type mismatch: GIS={gis['pole_type']} vs AI={ai['pole_type']}")
    # Compare conduit riser
    if str(ai["has_conduit_riser"]).strip().lower() != str(gis['has_conduit_riser']).strip().lower():
        mismatches.append(f"Conduit riser mismatch: GIS={gis['has_conduit_riser']} vs AI={ai['has_conduit_riser']}")
    return mismatches

def compute_risk(ai, mismatches):
    score = 100
    if str(ai.get("vegetation", "No")).strip().lower() == "yes":
        score -= 15
    if str(ai.get("guy_guard", "Yes")).strip().lower() == "no":
        score -= 35
    score -= 25 * len(mismatches)
    score = max(0, min(100, score))
    if score >= 80:
        status = "OK"
    elif score >= 50:
        status = "WARNING"
    else:
        status = "HIGH RISK"
    return score, status

def save_report(report, out_dir=REPORTS_DIR):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{report['pole_id']}_{ts}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path

# --------- UI ---------

st.title("Pole Review")
st.markdown("Compare GIS data with AI inspection results.")

# load the GIS data once
try:
    gis_df = load_gis(CSV_PATH)
except Exception as e:
    st.error(f"Failed to read GIS CSV at {CSV_PATH}: {e}")
    st.stop()

pole_list = list(gis_df["pole_id"].values)

# Get pole from session state (set by app.py button)
selected = st.session_state.get("selected_pole", "")

st.write(f"DEBUG - Selected pole from session: '{selected}'")

if not selected or selected not in pole_list:
    st.warning(f"No pole selected. Go back to the home page and select a pole to review.")
    st.stop()

st.success(f"Pole {selected} found! Proceeding...")

# Pole is confirmed to exist, proceed directly
gis_row = get_pole_record(gis_df, selected)

st.subheader("GIS record")
st.json(gis_row)

st.subheader("AI inspection (editable)")
veg = st.radio("Vegetation detected?", options=["Yes", "No"], index=0, key="veg")
gg = st.radio("Guy guard detected?", options=["Yes", "No"], index=1, key="gg")
ptype = st.selectbox("Pole type detected:", options=["Wood", "Steel"], index=0, key="ptype")
riser = st.radio("Conduit riser detected?", options=["Yes", "No"], index=1, key="riser")

if st.button("Run comparison"):
    ai_result = {
        "pole_id": selected,
        "vegetation": veg,
        "guy_guard": gg,
        "pole_type": ptype,
        "has_conduit_riser": riser,
    }
    
    # display records side-by-side
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("GIS record")
        st.json(gis_row)
    with col_b:
        st.subheader("AI result")
        st.json(ai_result)

    mismatches = compare(ai_result, gis_row)
    score, status = compute_risk(ai_result, mismatches)

    st.markdown("### Results")
    st.metric("Risk score", f"{score}")
    st.write("Status:", status)
    if mismatches:
        for m in mismatches:
            st.error(m)
    else:
        st.success("No mismatches detected.")
    
    # save report
    report = {
        "pole_id": selected,
        "timestamp": datetime.datetime.now().isoformat(),
        "gis": gis_row,
        "ai": ai_result,
        "mismatches": mismatches,
        "risk_score": score,
        "status": status
    }
    path = save_report(report)
    st.info(f"Report saved to {path}")