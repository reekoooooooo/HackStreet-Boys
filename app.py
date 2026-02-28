# app.py
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import pytesseract
import io
import os
import json
import datetime

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "gis_records.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
st.set_page_config(page_title="PolePad AI - Demo", layout="wide")

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

# ---------------------------
# Simple OCR helper (pole id suggestion)
# ---------------------------
def suggest_pole_id_from_image(pil_img):
    try:
        gray = pil_img.convert("L")
        text = pytesseract.image_to_string(gray, config="--psm 6")  # general block of text
        # simple heuristic: pick first token that looks like P followed by digits
        tokens = [t.strip() for t in text.replace("\n", " ").split(" ") if t.strip()]
        for tok in tokens:
            if tok.upper().startswith("P") and any(ch.isdigit() for ch in tok):
                # sanitize: keep only P and digits
                clean = "".join([c for c in tok if c.upper() == "P" or c.isdigit()])
                return clean.upper()
    except Exception:
        return None
    return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("PolePad AI — Verification Demo")
st.markdown("Upload a pole image (or pick a pole id) → simulate AI detection → compare against GIS → show mismatches & risk score.")

# Load GIS
try:
    gis_df = load_gis(CSV_PATH)
except Exception as e:
    st.error(f"Could not load GIS CSV at {CSV_PATH}: {e}")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload an inspection image (optional)", type=["jpg", "jpeg", "png"])
    st.write("Or select a pole from the GIS below:")
    pole_list = list(gis_df["pole_id"].values)
    selected_pole = st.selectbox("Select pole_id", options=[""] + pole_list, index=0)

    # OCR suggestion
    suggested = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        if st.button("Suggest pole_id from image (OCR)"):
            with st.spinner("Running OCR suggestion..."):
                suggested = suggest_pole_id_from_image(image)
                if suggested:
                    st.success(f"Suggested: {suggested}")
                else:
                    st.warning("No plausible pole_id found by OCR. You can type/paste it manually or pick from dropdown.")

    st.markdown("---")
    st.subheader("Simulate AI detection (editable)")
    # pre-fill Pole ID from selection or OCR suggestion
    initial_pole = selected_pole if selected_pole else (suggested or "")
    pole_input = st.text_input("pole_id (e.g., P0004)", value=initial_pole)

    veg = st.radio("Vegetation detected?", options=["Yes", "No"], index=0)
    gg = st.radio("Guy guard detected?", options=["Yes", "No"], index=1)
    ptype = st.selectbox("Pole type detected:", options=["Wood", "Steel"], index=0)
    riser = st.radio("Conduit riser detected?", options=["Yes", "No"], index=1)

    if st.button("Run verification"):
        if not pole_input:
            st.error("Please provide a pole_id (either select from dropdown or type/paste one).")
        else:
            ai_result = {
                "pole_id": pole_input.strip().upper(),
                "vegetation": veg,
                "guy_guard": gg,
                "pole_type": ptype,
                "has_conduit_riser": riser
            }

            gis_row = get_pole_record(gis_df, ai_result["pole_id"])
            if gis_row is None:
                st.error(f"No GIS record found for pole_id {ai_result['pole_id']}. You can pick a different pole or add a mock record to the CSV.")
            else:
                mismatches = compare(ai_result, gis_row)
                score, status = compute_risk(ai_result, mismatches)

                # build and save report
                report = {
                    "pole_id": ai_result["pole_id"],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "gis": gis_row,
                    "ai": ai_result,
                    "mismatches": mismatches,
                    "risk_score": score,
                    "status": status
                }
                out_path = save_report(report)

                # pass to UI area for display
                st.session_state["last_report"] = report
                st.session_state["last_report_path"] = out_path

with col2:
    st.header("Preview & Results")
    if uploaded:
        st.image(Image.open(uploaded).convert("RGB"), use_column_width=True)
    else:
        st.info("No image uploaded. Use the inputs on the left to simulate an AI detection.")

    last = st.session_state.get("last_report", None)
    if last:
        st.markdown("### Verification Results")
        st.metric("Risk Score", f"{last['risk_score']}", delta=None)
        st.write("**Status:**", last["status"])
        st.markdown("**GIS record**")
        st.json(last["gis"])
        st.markdown("**AI detected**")
        st.json(last["ai"])
        st.markdown("**Mismatches**")
        if last["mismatches"]:
            for m in last["mismatches"]:
                st.error(m)
        else:
            st.success("No mismatches found.")
        st.markdown("---")
        st.write(f"Report saved to `{st.session_state.get('last_report_path')}`")
    else:
        st.info("Run a verification to see results here.")

st.sidebar.markdown("## Notes")
st.sidebar.write("- GIS must be present as `gis_records.csv` in the project folder.")
st.sidebar.write("- Yes/No fields are case-insensitive.")
st.sidebar.write("- This simulates AI output; replace with real model outputs for production.")
st.sidebar.write("- Reports are saved to the `reports/` folder.")