# Pipeline Modes Summary

## 1. Roboflow Mode
**Purpose:**
- Detects general infrastructure elements: poles, wires, and vegetation in wide-area images.

**When to Use:**
- Use for images showing entire utility poles, wires, and surrounding vegetation (not close-ups of tags or numbers).

**Command:**
```
python src/run_all_pipeline.py path/to/image.jpg --roboflow
```

**Outputs:**
- **Terminal:**
  - Prints the number and type of detected objects (e.g., poles, wires, vegetation).
  - Shows a vegetation score (1-100).
- **Visualization Image:**
  - Saved to: `roboflow_visualizations/<image_name>_roboflow.png`
  - Shows detected objects with bounding boxes.
- **CSV Summary:**
  - Appends a row to: `roboflow_output/roboflow_results.csv`
  - Columns: image, wire_count, vegetation_score

---

## 2. Normal Mode (OCR + YOLOv8 Tag Detection)
**Purpose:**
- Extracts pole tag numbers and asset information from close-up images.

**When to Use:**
- Use for images focused on pole tags, numbers, or asset labels (close-up shots).

**Command:**
```
python src/run_all_pipeline.py path/to/image.jpg
```

**Outputs:**
- **Terminal:**
  - Prints OCR results (detected text/numbers).
  - Prints GIS attributes (pole ID, type, vegetation encroachment, etc.).
- **CSV Output:**
  - Appends a row to: `output_gis.csv`
  - Columns: pole_id, pole_type, vegetation_encroachment, from_ocr

---

## Summary Table

| Mode         | Command                                               | Use For                        | Main Outputs                                                                                   |
|--------------|-------------------------------------------------------|--------------------------------|-----------------------------------------------------------------------------------------------|
| Roboflow     | python src/run_all_pipeline.py path/to/image.jpg --roboflow | Wide shots (poles, wires, veg) | Terminal summary, visualization image (roboflow_visualizations/), roboflow_results.csv         |
| Normal (OCR) | python src/run_all_pipeline.py path/to/image.jpg      | Close-ups (tags, numbers)      | Terminal OCR & GIS info, output_gis.csv                                                       |

---

**Tip:**
- Always check the output folders for your results:
  - roboflow_visualizations/ for images with detection boxes
  - roboflow_output/roboflow_results.csv for Roboflow summaries
  - output_gis.csv for OCR/tag extraction results
