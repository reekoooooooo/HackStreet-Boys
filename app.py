# app.py - Main Application
import streamlit as st
import pandas as pd
from PIL import Image
import os
import tempfile
import sys

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "gis_records.csv"
st.set_page_config(page_title="PolePad AI", layout="wide")
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ---------------------------
# Utility functions
# ---------------------------
def load_gis(path=CSV_PATH):
    df = pd.read_csv(path)
    return df

# ---------------------------
# Streamlit UI
# ---------------------------
st.title(":blue[PolePad AI] — Utility inspection made easy")
st.subheader("Defects flagged by AI, Reviewed by :blue[You]")

# Load GIS
try:
    gis_df = load_gis(CSV_PATH)
except Exception as e:
    st.error(f"Could not load GIS CSV at {CSV_PATH}: {e}")
    st.stop()

st.header(":blue[New Inspection]")
st.markdown("Upload a pole image → analyze with AI and compare with GIS database.")
    
mode = st.radio(
"Choose analysis mode:",
[
    "Tag Extraction (OCR, close-up tags/numbers)",
    "Scene Detection (Poles, Wires, Vegetation, etc. - Roboflow)"
]
)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        temp_path = tmp.name
    
    # Store original filename (without extension) as pole_id
    original_filename = os.path.splitext(uploaded_file.name)[0]

    if st.button("Analyze"):
        if "Tag Extraction" in mode:
            try:
                from run_all_pipeline import run_all, compare_ocr_with_gis_records_ocr
            except Exception as e:
                st.error(f"Unable to import pipeline: {e}")
            else:
                with st.spinner("Running OCR + GIS pipeline..."):
                    try:
                        res = run_all(temp_path, pole_id=original_filename)
                        st.success("Pipeline finished")
                        st.write("### Pipeline Results")
                        st.json(res['gis_attributes'])
                        
                        # Get pole_id from the uploaded image (temp filename)
                        uploaded_pole_id = res['gis_attributes'].get('pole_id')
                        
                        # Run comparison for only this pole
                        st.subheader("Comparison with GIS Database")
                        comparisons = compare_ocr_with_gis_records_ocr(pole_id_filter=uploaded_pole_id)
                        if comparisons and 'error' in comparisons[0]:
                            st.info(f"Pole '{uploaded_pole_id}' not found in GIS database")
                        elif not comparisons:
                            st.success(f"✅ Pole '{uploaded_pole_id}' matches GIS database - no mismatches found")
                        else:
                            for comp in comparisons:
                                pole_id = comp['pole_id']
                                st.warning(f"**Pole {pole_id}: Mismatches Found**")
                                for m in comp['mismatches']:
                                    st.error(f"  • {m}")
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
        else:
            try:
                from run_all_pipeline import run_roboflow_inference, compare_roboflow_with_gis_records
            except Exception as e:
                st.error(f"Unable to import pipeline: {e}")
            else:
                with st.spinner("Running Roboflow inference..."):
                    try:
                        res = run_roboflow_inference(temp_path, pole_id=original_filename)
                        st.success("Roboflow inference finished")
                        st.write("### Roboflow Results")
                        st.write(f"**Detections:** {len(res.get('detections', []))}")
                        st.write(f"**Wire Count:** {res.get('wire_count', 0)}")
                        st.write(f"**Vegetation Score:** {res.get('vegetation_score', 0)}")
                        if res.get('visualization'):
                            st.image(res['visualization'], caption="Roboflow Visualization")
                        
                        # Get pole_id from the uploaded image
                        uploaded_pole_id = res.get('pole_id')
                        
                        # Run comparison for only this pole
                        st.subheader("Comparison with GIS Database")
                        comparisons = compare_roboflow_with_gis_records(pole_id_filter=uploaded_pole_id)
                        if comparisons and 'error' in comparisons[0]:
                            st.info(f"Pole '{uploaded_pole_id}' not found in GIS database")
                        elif not comparisons:
                            st.success(f"✅ Pole '{uploaded_pole_id}' matches GIS database - no mismatches found")
                        else:
                            for comp in comparisons:
                                pole_id = comp['pole_id']
                                st.warning(f"**Pole {pole_id}: Mismatches Found**")
                                for m in comp['mismatches']:
                                    st.error(f"  • {m}")
                    except Exception as e:
                        st.error(f"Roboflow error: {e}")

st.sidebar.markdown("## Notes")
st.sidebar.write("- Upload a pole image to analyze with AI")
st.sidebar.write("- Results are automatically compared with GIS database")
st.sidebar.write("- Original filename is used as pole_id")
