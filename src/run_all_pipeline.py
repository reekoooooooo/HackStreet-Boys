import os
import cv2
import base64
import csv
import pandas as pd
from preprocess import preprocess

def run_all(image_path, yolo_model_path='yolov8s.pt', output_csv='output_gis.csv', pole_id=None):
    """
    Given an original image path, preprocesses the image, runs OCR, and GIS detection.
    Returns OCR results and GIS attributes. Also writes GIS attributes to CSV.
    pole_id: Optional pole identifier to use instead of deriving from filename.
    """
    # Lazy import heavy dependencies
    try:
        import easyocr
    except Exception as e:
        raise ImportError(f"Missing dependency 'easyocr': {e}")
    
    try:
        from infra_gis_detect import detect_infrastructure_attributes, write_gis_csv
    except Exception as e:
        raise ImportError(f"Could not import infra_gis_detect: {e}")
    
    # 1. Preprocess
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    processed_img = preprocess(img)
    # Save processed image
    processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    base, ext = os.path.splitext(os.path.basename(image_path))
    processed_filename = f"{base}_processed{ext}"
    processed_path = os.path.join(processed_dir, processed_filename)
    cv2.imwrite(processed_path, processed_img)

    # 2. OCR on processed image
    reader = easyocr.Reader(['en'])
    ocr_results = reader.readtext(processed_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    ocr_texts = []
    for _, text, conf in ocr_results:
        filtered = ''.join([c for c in text if c.isalnum()])
        if filtered:
            ocr_texts.append({'text': filtered, 'confidence': conf})

    # 3. GIS detection (YOLO+EasyOCR)
    gis_attributes = detect_infrastructure_attributes(processed_path, yolo_model_path, pole_id=pole_id)
    # Write GIS attributes to CSV
    write_gis_csv(gis_attributes, output_csv)

    return {
        'processed_image': processed_path,
        'ocr_results': ocr_texts,
        'gis_attributes': gis_attributes,
        'csv': output_csv
    }

def compare_ocr_with_gis_records_ocr(output_gis_csv='output_gis.csv', gis_records_ocr_csv='gis_records_ocr.csv', pole_id_filter=None):
    """
    Compare OCR AI findings (output_gis.csv) with GIS database (gis_records_ocr.csv).
    Only compares poles that have matching pole_id in both datasets.
    If pole_id_filter is provided, only compares that specific pole.
    Returns list of mismatches per pole.
    """
    results = []
    try:
        ai_df = pd.read_csv(output_gis_csv)
        gis_df = pd.read_csv(gis_records_ocr_csv)
    except FileNotFoundError as e:
        return [{"error": f"CSV file not found: {e}"}]
    
    # Filter to specific pole if requested
    if pole_id_filter:
        ai_df = ai_df[ai_df['pole_id'] == pole_id_filter]
        if ai_df.empty:
            return [{"error": f"Pole {pole_id_filter} not found in AI results"}]
    
    for _, ai_row in ai_df.iterrows():
        pole_id = ai_row.get('pole_id')
        gis_match = gis_df[gis_df['pole_id'] == pole_id]
        
        # Skip if no matching record in GIS database
        if gis_match.empty:
            continue
        
        gis_row = gis_match.iloc[0]
        mismatches = []
        
        # Compare fields only if GIS has a non-null value
        gis_tag = gis_row.get('tag_name')
        gis_type = gis_row.get('pole_type')
        gis_veg = gis_row.get('vegetation_encroachment')
        
        if pd.notna(gis_tag) and str(ai_row.get('tag_name', '')).strip() != str(gis_tag).strip():
            mismatches.append(f"Tag name mismatch: GIS={gis_tag} vs AI={ai_row.get('tag_name')}")
        if pd.notna(gis_type) and str(ai_row.get('pole_type', '')).strip().lower() != str(gis_type).strip().lower():
            mismatches.append(f"Pole type mismatch: GIS={gis_type} vs AI={ai_row.get('pole_type')}")
        if pd.notna(gis_veg) and str(ai_row.get('vegetation_encroachment', '')).strip().lower() != str(gis_veg).strip().lower():
            mismatches.append(f"Vegetation mismatch: GIS={gis_veg} vs AI={ai_row.get('vegetation_encroachment')}")
        
        # Only add to results if there are actual mismatches
        if mismatches:
            results.append({
                'pole_id': pole_id,
                'mismatches': mismatches,
                'ai_data': ai_row.to_dict(),
                'gis_data': gis_row.to_dict()
            })
    
    return results

def compare_roboflow_with_gis_records(roboflow_csv='roboflow_output/roboflow_results.csv', gis_records_csv='gis_records.csv', pole_id_filter=None):
    """
    Compare Roboflow AI findings (roboflow_results.csv) with GIS database (gis_records.csv).
    Only compares poles that have matching pole_id in both datasets.
    If pole_id_filter is provided, only compares that specific pole.
    Returns list of mismatches per pole.
    """
    results = []
    try:
        ai_df = pd.read_csv(roboflow_csv)
        gis_df = pd.read_csv(gis_records_csv)
    except FileNotFoundError as e:
        return [{"error": f"CSV file not found: {e}"}]
    
    # Filter to specific pole if requested
    if pole_id_filter:
        ai_df = ai_df[ai_df['pole_id'] == pole_id_filter]
        if ai_df.empty:
            return [{"error": f"Pole {pole_id_filter} not found in AI results"}]
    
    for _, ai_row in ai_df.iterrows():
        pole_id = ai_row.get('pole_id')
        gis_match = gis_df[gis_df['pole_id'] == pole_id]
        
        # Skip if no matching record in GIS database
        if gis_match.empty:
            continue
        
        gis_row = gis_match.iloc[0]
        mismatches = []
        
        # Compare fields only if GIS has a non-null value
        gis_wire = gis_row.get('wire_count')
        gis_veg = gis_row.get('vegetation_score')
        
        if pd.notna(gis_wire) and int(ai_row.get('wire_count', 0)) != int(gis_wire):
            mismatches.append(f"Wire count mismatch: GIS={int(gis_wire)} vs AI={ai_row.get('wire_count')}")
        if pd.notna(gis_veg) and int(ai_row.get('vegetation_score', 0)) != int(gis_veg):
            mismatches.append(f"Vegetation score mismatch: GIS={int(gis_veg)} vs AI={ai_row.get('vegetation_score')}")
        
        # Only add to results if there are actual mismatches
        if mismatches:
            results.append({
                'pole_id': pole_id,
                'mismatches': mismatches,
                'ai_data': ai_row.to_dict(),
                'gis_data': gis_row.to_dict()
            })
    
    return results

def run_roboflow_inference(image_path, pole_id=None):
    """
    Runs Roboflow inference on the given image and prints the results.
    Also saves the visualization image to disk in a new folder and saves results to CSV.
    pole_id: Optional pole identifier to use instead of deriving from filename.
    """
    # Lazy import heavy dependencies
    try:
        from inference_sdk import InferenceHTTPClient
    except Exception as e:
        raise ImportError(f"Missing dependency 'inference_sdk': {e}")
    
    import base64
    
    # Replace with your actual API key from Roboflow dashboard if needed
    api_key = "XrAdoF6M9mxcYRG0qyLi"  # Find this in Roboflow: Workspace > Settings > API Key
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )
    result = client.run_workflow(
        workspace_name="polepad",
        workflow_id="find-poles-wires-and-vegetations",
        images={
            "image": image_path
        },
        use_cache=True
    )
    # Extract predictions
    preds = []
    if isinstance(result, list) and result:
        preds = result[0].get('predictions', {}).get('predictions', [])
        print(f"Roboflow inference: {len(preds)} objects detected.")
        for pred in preds:
            print(f"  Class: {pred.get('class', 'unknown')}, Confidence: {pred.get('confidence', 0):.2f}")
    else:
        print("Roboflow inference: No result or unexpected format.")

    # Count wires and vegetation
    wire_count = sum(1 for p in preds if p.get('class', '').lower() in ['wire', 'wires', 'line', 'power line'])
    veg_count = sum(1 for p in preds if 'vegetation' in p.get('class', '').lower())
    
    # Vegetation score: scale, then cut by 50% but max 100
    raw_score = min(100, veg_count * 20) if veg_count > 0 else 0
    vegetation_score = min(100, int(raw_score * 0.5))

    # Save visualization image to disk
    vis_b64 = result[0].get('visualization') if isinstance(result, list) and result else None
    vis_path = None
    if vis_b64:
        vis_folder = os.path.join(os.path.dirname(__file__), 'roboflow_visualizations')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(vis_folder, f"{base_name}_roboflow.png")
        with open(vis_path, "wb") as f:
            f.write(base64.b64decode(vis_b64))
        print(f"Visualization image saved to: {vis_path}")
    else:
        print("No visualization image found in Roboflow result.")
    
    # Extract pole_id from filename if not provided
    if pole_id is None:
        pole_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save to CSV
    csv_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'roboflow_output')
    csv_path = save_roboflow_csv(pole_id, wire_count, vegetation_score, csv_folder=csv_folder)
    
    # Return results
    return {
        'pole_id': pole_id,
        'visualization': vis_path,
        'detections': preds,
        'predictions': result,
        'wire_count': wire_count,
        'vegetation_score': vegetation_score,
        'csv_path': csv_path
    }

def save_roboflow_csv(pole_id, wire_count, vegetation_score, csv_folder='roboflow_output'):
    """
    Save Roboflow summary results to a CSV file in the specified folder.
    Updates existing entry if pole_id already exists, otherwise appends new entry.
    """
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    csv_path = os.path.join(csv_folder, 'roboflow_results.csv')
    
    # Read existing CSV if it exists
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        # Update existing entry or append new one
        if pole_id in df['pole_id'].values:
            df.loc[df['pole_id'] == pole_id, ['wire_count', 'vegetation_score']] = [wire_count, vegetation_score]
        else:
            new_row = pd.DataFrame([{
                'pole_id': pole_id,
                'image': f'{pole_id}.jpg',
                'wire_count': wire_count,
                'vegetation_score': vegetation_score
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
    else:
        # Create new CSV
        df = pd.DataFrame([{
            'pole_id': pole_id,
            'image': f'{pole_id}.jpg',
            'wire_count': wire_count,
            'vegetation_score': vegetation_score
        }])
        df.to_csv(csv_path, index=False)
    
    return csv_path

