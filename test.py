import streamlit as st
from PIL import Image
import os
import tempfile
import subprocess


st.title("PolePad AI: Infrastructure Image Analyzer")


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
    st.image(img, caption="Uploaded Image", use_column_width=True)
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        temp_path = tmp.name


    if st.button("Analyze"):
        if "Tag Extraction" in mode:
            cmd = f"python src/run_all_pipeline.py {temp_path}"
        else:
            cmd = f"python src/run_all_pipeline.py {temp_path} --roboflow"
        st.write(f"Running: `{cmd}`")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)