# Group Name: The HackStreet Boys

## Members: David Ringler, Tyrek Long, Pranav Kanumuri, Raymart Velasco, Alex Santiago.

## Initial Plan:


# PolePad AI: Crowd-Powered Infrastructure Verification

## Team: The HackStreet Boys
**Members:** David Ringler, Tyrek Long, Pranav Kanumuri, Raymart Velesco, Alex Santiago

---

## Project Overview
PolePad AI is an AI-powered system for analyzing utility pole inspection images and turning them into structured, verifiable infrastructure data. It supports both automated detection and human-in-the-loop validation, enabling distributed, crowd-powered asset verification for utilities like Dominion Energy.

---

## Features
- **Two Pipeline Modes:**
	- **Tag Extraction (OCR):** For close-up images of pole tags, numbers, or asset labels. Extracts alphanumeric IDs and attributes.
	- **Scene Detection (Roboflow):** For wide shots of poles, wires, and vegetation. Detects infrastructure elements and computes vegetation risk.
- **Streamlit Web UI:** Simple interface for uploading images and choosing analysis mode.
- **Confidence & Review Columns:** All outputs include confidence scores, review-needed flags, and consensus tracking for human validation.
- **CSV & Visualization Outputs:** Results are saved as CSVs for easy integration, and annotated images are saved for review.

---

## How to Run

### 1. Command Line
- **Tag Extraction (OCR):**
	```
	python src/run_all_pipeline.py path/to/image.jpg
	```
- **Scene Detection (Roboflow):**
	```
	python src/run_all_pipeline.py path/to/image.jpg --roboflow
	```

### 2. Streamlit Web App
```
streamlit run app_streamlit.py
```
Upload an image and select the mode in the browser.

---

## Outputs
- **CSV Files:**
	- `output_gis.csv` (for OCR/tag mode): Includes pole_id, type, vegetation, confidence, review_needed, validated, consensus.
	- `roboflow_output/roboflow_results.csv` (for Roboflow mode): Includes image, wire_count, vegetation_score.
- **Visualization Images:**
	- Annotated images saved in `roboflow_visualizations/`.

---

## Tech Stack
- **Python** (core language)
- **OpenCV** (image preprocessing)
- **EasyOCR** (text/number extraction)
- **YOLOv8** (object detection)
- **Roboflow Inference SDK** (cloud model integration)
- **Streamlit** (web UI)
- **CSV** (structured outputs)

---

## Value Proposition
- **Reduces manual data entry and errors**
- **Accelerates field audits and asset verification**
- **Enables distributed, crowd-powered validation**
- **Creates a living, continuously updated asset registry**
- **Supports both AI automation and human expertise**

---

## For Presentations
- See `docs/pipeline_modes_summary.md` for a ready-to-use summary.
- Example Q&A and tech stack breakdown included in project documentation.
