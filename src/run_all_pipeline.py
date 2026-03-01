import os
import cv2
import base64
import csv
from preprocess import preprocess
from infra_gis_detect import detect_infrastructure_attributes
import easyocr
from inference_sdk import InferenceHTTPClient

def run_all(image_path, yolo_model_path='yolov8s.pt', output_csv='output_gis.csv'):
    """
    Given an original image path, preprocesses the image, runs OCR, and GIS detection.
    Returns OCR results and GIS attributes. Also writes GIS attributes to CSV.
    """
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
    gis_attributes = detect_infrastructure_attributes(processed_path, yolo_model_path)
    # Write GIS attributes to CSV
    from infra_gis_detect import write_gis_csv
    write_gis_csv(gis_attributes, output_csv)

    return {
        'processed_image': processed_path,
        'ocr_results': ocr_texts,
        'gis_attributes': gis_attributes,
        'csv': output_csv
    }

def run_roboflow_inference(image_path):
    """
    Runs Roboflow inference on the given image and prints the results.
    Also saves the visualization image to disk in a new folder.
    """
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
    # Print only summary of predictions, not the full base64 visualization
    if isinstance(result, list) and result:
        preds = result[0].get('predictions', {}).get('predictions', [])
        print(f"Roboflow inference: {len(preds)} objects detected.")
        for pred in preds:
            print(f"  Class: {pred.get('class', 'unknown')}, Confidence: {pred.get('confidence', 0):.2f}")
    else:
        print("Roboflow inference: No result or unexpected format.")

    # Save visualization image to disk
    vis_b64 = result[0].get('visualization') if isinstance(result, list) and result else None
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

def save_roboflow_csv(image_path, wire_count, vegetation_score, csv_folder='roboflow_output'):
    """
    Save Roboflow summary results to a CSV file in the specified folder.
    """
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    csv_path = os.path.join(csv_folder, 'roboflow_results.csv')
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image', 'wire_count', 'vegetation_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'image': os.path.basename(image_path),
            'wire_count': wire_count,
            'vegetation_score': vegetation_score
        })
    return csv_path

def process_image(image_path, mode='ocr_gis'):
    """
    Unified entry point for UI or CLI.
    mode: 'roboflow' for Roboflow workflow, 'ocr_gis' for OCR+YOLOv8 GIS workflow.
    Returns a dictionary with results and output paths.
    """
    if mode == 'roboflow':
        # Roboflow inference
        from inference_sdk import InferenceHTTPClient
        import base64
        api_key = "XrAdoF6M9mxcYRG0qyLi"
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        # Debug: print and check image path
        print(f"[DEBUG] Roboflow image path: {image_path}")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        result = client.run_workflow(
            workspace_name="polepad",
            workflow_id="find-poles-wires-and-vegetations",
            images={"image": image_path},
            use_cache=True
        )
        # Save visualization (fix: always use input image base name)
        vis_b64 = result[0].get('visualization') if isinstance(result, list) and result else None
        vis_path = None
        if vis_b64:
            # Always save to project root's roboflow_visualizations
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            vis_folder = os.path.join(project_root, 'roboflow_visualizations')
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            vis_path = os.path.join(vis_folder, f"{base_name}_roboflow.png")
            with open(vis_path, "wb") as f:
                f.write(base64.b64decode(vis_b64))
        # Parse predictions
        preds = result[0].get('predictions', {}).get('predictions', []) if isinstance(result, list) and result else []
        # Count wires and vegetation
        wire_count = sum(1 for p in preds if p.get('class', '').lower() in ['wire', 'wires', 'line', 'power line'])
        veg_count = sum(1 for p in preds if 'vegetation' in p.get('class', '').lower())
        # Vegetation score: scale, then cut by 50% but max 100
        raw_score = min(100, veg_count * 20) if veg_count > 0 else 0
        vegetation_score = min(100, int(raw_score * 0.5))
        # Save CSV summary in project root
        csv_path = save_roboflow_csv(image_path, wire_count, vegetation_score, csv_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'roboflow_output'))
        return {
            'mode': 'roboflow',
            'detections': preds,
            'visualization': vis_path,
            'raw_result': result,
            'wire_count': wire_count,
            'vegetation_score': vegetation_score,
            'csv': csv_path
        }
    else:
        # OCR + GIS pipeline
        res = run_all(image_path)
        return {
            'mode': 'ocr_gis',
            'ocr_results': res['ocr_results'],
            'gis_attributes': res['gis_attributes'],
            'processed_image': res['processed_image'],
            'csv': res['csv']
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_all_pipeline.py <original_image_path> [--roboflow]")
        sys.exit(1)
    image_path = sys.argv[1]
    mode = 'roboflow' if (len(sys.argv) > 2 and sys.argv[2] == "--roboflow") else 'ocr_gis'
    result = process_image(image_path, mode)
    print(f"\n--- Pipeline Mode: {result['mode']} ---")
    if result['mode'] == 'roboflow':
        print(f"Visualization image: {result['visualization']}")
        print(f"Detections: {len(result['detections'])}")
        print(f"Wire count: {result['wire_count']}")
        print(f"Vegetation score (1-100): {result['vegetation_score']}")
        print(f"Roboflow CSV updated: {result['csv']}")
        for pred in result['detections']:
            print(f"  Class: {pred.get('class', 'unknown')}, Confidence: {pred.get('confidence', 0):.2f}")
    else:
        print(f"Processed image: {result['processed_image']}")
        print("OCR Results:")
        for r in result['ocr_results']:
            print(f"  {r['text']} (Confidence: {r['confidence']:.2f})")
        print("GIS Attributes:")
        for k, v in result['gis_attributes'].items():
            print(f"  {k}: {v}")
        print(f"GIS CSV updated: {result['csv']}")

    # Example usage:
    # run_roboflow_inference("src/images/PoleTag_24.jpg")
    if __name__ == "__main__":
        run_roboflow_inference("src/images/PoleTag_24.jpg")

    # --- How to find your Roboflow API key ---
    # 1. Log in to https://roboflow.com/
    # 2. Go to your workspace (top left dropdown)
    # 3. Click "Settings"
    # 4. Scroll to "API Key" section
    # 5. Copy your API key and paste it above if needed