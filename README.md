# PolePad AI - Utility Pole Inspection System

**Team:** The HackStreet Boys (David Ringler, Tyrek Long, Pranav Kanumuri, Raymart Velasco, Alex Santiago)

## Overview

PolePad AI is an automated utility pole inspection system that combines computer vision (YOLO object detection + OCR) and cloud-based detection (Roboflow) to analyze pole images and compare results against a GIS database.

## Features

### 1. **Dual Analysis Modes**
   - **Tag Extraction (OCR + YOLO)**: Detects serial numbers, tags, and infrastructure attributes using EasyOCR and YOLOv8
   - **Scene Detection (Roboflow)**: Cloud-based detection for poles, wires, vegetation, and other features

### 2. **Automated Comparison**
   - Compares AI-detected attributes against GIS database records
   - Identifies mismatches in:
     - Pole tags/IDs (for OCR mode)
     - Wire counts (for Roboflow mode)
     - Vegetation encroachment/scores
     - Pole types

### 3. **Persistent Tracking**
   - Original filename used as pole_id (prevents duplicate entries)
   - CSV updates existing records instead of creating duplicates
   - Maintains separate output for OCR and Roboflow results

## Installation

### Requirements
- Python 3.8+
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd HackStreet-Boys

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (first run only)
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

## Usage

### Running the Application

```bash
cd HackStreet-Boys
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Uploading an Image

1. Select analysis mode (Tag Extraction or Scene Detection)
2. Upload a pole image (JPG, PNG, WEBP)
3. Click "Analyze" button
4. View:
   - Detection results
   - GIS database comparison
   - Mismatch details (if any)

### Important Notes
- Use original filename (without extension) as pole_id for proper tracking
- Requires internet connection for Roboflow inference
- Roboflow API key is configured in `src/run_all_pipeline.py`

## Project Structure

```
HackStreet-Boys/
├── app.py                          # Streamlit UI application
├── requirements.txt                # Python dependencies
├── gis_records.csv                 # GIS database (poles/wires/vegetation)
├── gis_records_ocr.csv            # OCR ground truth data
├── output_gis.csv                  # OCR pipeline results
│
├── src/
│   ├── run_all_pipeline.py        # Core pipeline orchestration
│   ├── infra_gis_detect.py        # YOLO + OCR detection
│   ├── preprocess.py               # Image preprocessing (contrast/sharpening)
│   ├── requirements.txt            # Development reference
│   ├── processed/                  # Preprocessed images
│   └── roboflow_visualizations/   # Output visualizations
│
└── roboflow_output/
    └── roboflow_results.csv       # Roboflow pipeline results
```

## Data Flow

### OCR + YOLO Pipeline
```
User Image → Preprocessing → OCR Detection → YOLO Detection 
→ output_gis.csv → GIS Comparison → Results
```

### Roboflow Pipeline
```
User Image → Roboflow Cloud API → Detection Results 
→ roboflow_results.csv → GIS Comparison → Results
```

## CSV Formats

### Input: gis_records.csv
```
pole_id, image, wire_count, vegetation_score
```

### Output: output_gis.csv
```
pole_id, tag_name, pole_type, vegetation_encroachment, from_ocr
```

### Output: roboflow_results.csv
```
pole_id, image, wire_count, vegetation_score
```

### Input: gis_records_ocr.csv
```
pole_id, tag_name, pole_type, vegetation_encroachment
```

## Key Components

### Image Preprocessing (`src/preprocess.py`)
- Resizes large images (max 1024px)
- Enhances contrast using CLAHE
- Applies sharpening filter

### YOLO Detection (`src/infra_gis_detect.py`)
- Detects infrastructure using YOLOv8
- Runs OCR on detected regions
- Extracts pole_id from filename
- Outputs structured attributes

### Pipeline Orchestration (`src/run_all_pipeline.py`)
- `run_all()`: OCR + YOLO pipeline
- `run_roboflow_inference()`: Cloud detection
- `compare_ocr_with_gis_records_ocr()`: Compares OCR results
- `compare_roboflow_with_gis_records()`: Compares Roboflow results
- Lazy imports for optional dependencies

## Development Notes

### Lazy Import Pattern
Heavy ML packages (easyocr, ultralytics, inference_sdk) use lazy imports to prevent startup errors when packages are missing. They're only imported when specific functions are called.

### Pole ID Tracking
- Uses original uploaded filename (without extension) as stable pole_id
- Prevents duplicate entries when re-analyzing same pole
- Updates existing CSV records instead of appending

### Comparison Logic
Only compares poles that:
- Exist in both AI results and GIS database
- Have valid pole_id matches
- Skip null GIS values to prevent false mismatches

## Configuration

### Roboflow Setup (Optional)
To enable Roboflow detection:
1. Create account at https://roboflow.com/
2. Copy API key from Workspace → Settings → API Key
3. Update in `src/run_all_pipeline.py`:
   ```python
   api_key = "YOUR_API_KEY_HERE"
   ```

### GIS Database
Place `gis_records.csv` in project root with required columns:
- pole_id
- image
- wire_count  
- vegetation_score

## Performance

- YOLO detection: ~100-500ms per image
- OCR processing: ~500-2000ms per image (depends on text complexity)
- Roboflow inference: ~1-3s per image (cloud API dependent)

**Total analysis time: 2-5 seconds per image**

## Troubleshooting

### ModuleNotFoundError
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated

### YOLO Model Not Found
- Run: `python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"`
- Model will auto-download on first use

### Roboflow Connection Issues
- Check internet connection
- Verify API key in `src/run_all_pipeline.py`
- Workspace name: "polepad"
- Workflow ID: "find-poles-wires-and-vegetations"

### Out of Memory
- Reduce image resolution before upload
- Use server with 8GB+ RAM for processing

## Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Export reports in PDF format  
- [ ] Real-time video stream processing
- [ ] Mobile app version
- [ ] Integration with ArcGIS for live GIS updates
- [ ] Risk scoring based on detected anomalies

## License

Project developed as hackathon submission.

## Contact

For questions or contributions, contact the HackStreet Boys team.
