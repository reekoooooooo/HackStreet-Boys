import os
import cv2
from preprocess import preprocess
from infra_gis_detect import detect_infrastructure_attributes
import easyocr

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_all_pipeline.py <original_image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    result = run_all(image_path)
    print("Processed image saved at:", result['processed_image'])
    print("\nOCR Results:")
    for r in result['ocr_results']:
        print(f"  {r['text']} (Confidence: {r['confidence']:.2f})")
    print("\nGIS Attributes:")
    for k, v in result['gis_attributes'].items():
        print(f"  {k}: {v}")
    print(f"\nGIS CSV updated: {result['csv']}")