import os
import cv2
import csv

def detect_infrastructure_attributes(image_path, yolo_model_path='yolov8s.pt', pole_id=None):
    """
    Detects infrastructure attributes from an image using YOLO and EasyOCR.
    Returns a dictionary with detected attributes.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Lazy-import heavy dependencies so module import doesn't fail when packages are absent
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError(
            "Missing dependency 'ultralytics'. Install it with `pip install ultralytics` "
            "to enable YOLO-based detection. Original error: " + str(e)
        )

    try:
        import easyocr
    except Exception as e:
        raise ImportError(
            "Missing dependency 'easyocr'. Install it with `pip install easyocr` "
            "to enable OCR on detected regions. Original error: " + str(e)
        )

    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)
    results = yolo_model(img)

    # Use provided pole_id or extract from filename (without extension)
    if pole_id is None:
        pole_id = os.path.splitext(os.path.basename(image_path))[0]

    # Initialize attributes with more detail
    attributes = {
        'pole_id': pole_id,
        'tag_name': '',
        'pole_type': '',
        'vegetation_encroachment': False,
        'from_ocr': ''
    }

    # Use EasyOCR for pole ID (on all detected regions)
    reader = easyocr.Reader(['en'])
    best_id = ''
    best_id_conf = 0.0
    ocr_candidates = []
    # Accumulate all OCR results from all boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        ocr_results = reader.readtext(crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        for _, text, conf in ocr_results:
            filtered = ''.join([c for c in text if c.isalnum() or c == '-'])
            if len(filtered) >= 4:
                ocr_candidates.append((filtered, conf))
            if len(filtered) >= 4 and conf > best_id_conf and (filtered.isdigit() or '-' in filtered or filtered.isalnum()):
                best_id = filtered
                best_id_conf = conf
    # After all boxes, choose the longest filtered OCR result from all candidates
    if ocr_candidates:
        # If multiple have the same length, pick the one with highest confidence
        maxlen = max(len(x[0]) for x in ocr_candidates)
        longest = max([x for x in ocr_candidates if len(x[0]) == maxlen], key=lambda x: x[1])
        attributes['from_ocr'] = f"{longest[0]} (Conf: {longest[1]:.2f})"
        attributes['tag_name'] = longest[0]  # OCR-scanned tag name
    else:
        attributes['from_ocr'] = ''
        attributes['tag_name'] = ''

    # YOLO class mapping (example, you should update with your custom model/classes)
    class_map = yolo_model.names if hasattr(yolo_model, 'names') else {}
    for box in results[0].boxes:
        cls = int(box.cls[0]) if hasattr(box, 'cls') else None
        label = class_map.get(cls, str(cls))
        # Map all detected types to 'wood' or 'metal' only
        mapped_type = ''
        if label.lower() in ['wood', 'wooden', 'composite']:
            mapped_type = 'wood'
        elif label.lower() in ['steel', 'metal']:
            mapped_type = 'metal'
        if mapped_type and not attributes['pole_type']:
            attributes['pole_type'] = mapped_type
        if 'vegetation' in label.lower():
            attributes['vegetation_encroachment'] = True

    # Default to 'wood' if not detected
    if not attributes['pole_type']:
        attributes['pole_type'] = 'wood'
    return attributes

def write_gis_csv(attributes, output_csv):
    """
    Writes the detected attributes to a CSV file in GIS format.
    Updates existing entry if pole_id already exists, otherwise appends new entry.
    """
    import pandas as pd
    fieldnames = [
        'pole_id','tag_name', 'pole_type', 'vegetation_encroachment', 'from_ocr'
    ]
    
    pole_id = attributes.get('pole_id')
    
    # Read existing CSV if it exists
    if os.path.isfile(output_csv):
        df = pd.read_csv(output_csv)
        # Update existing entry or append new one
        if pole_id and pole_id in df['pole_id'].values:
            for field in fieldnames:
                if field in attributes:
                    df.loc[df['pole_id'] == pole_id, field] = attributes[field]
        else:
            row = {k: attributes.get(k, '') for k in fieldnames}
            new_row = pd.DataFrame([row])
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(output_csv, index=False)
    else:
        # Create new CSV
        row = {k: attributes.get(k, '') for k in fieldnames}
        df = pd.DataFrame([row])
        df.to_csv(output_csv, index=False)

