import sys
import os
import cv2
import easyocr
from ultralytics import YOLO
from preprocess import preprocess

def main():

    if len(sys.argv) < 2:
        print("Usage: python ocr_reader.py <original_image_name>")
        sys.exit(1)

    orig_path = sys.argv[1]
    base, ext = os.path.splitext(os.path.basename(orig_path))
    processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
    processed_filename = f"{base}_processed{ext}"
    processed_path = os.path.join(processed_dir, processed_filename)
    print(f"Loading processed image from: {processed_path}")
    img = cv2.imread(processed_path)
    if img is None:
        print("ERROR: Could not load processed image. Make sure to run preprocess.py first.")
        sys.exit(1)

    print("Running YOLO to detect text regions...")
    img = cv2.imread(processed_path)
    if img is None:
        print("ERROR: Could not load processed image.")
        sys.exit(1)

    # Load YOLO model (using pretrained YOLOv8s for text detection)
    yolo_model = YOLO('yolov8s.pt')
    results = yolo_model(img)
    print("Detected regions:", len(results[0].boxes))

    reader = easyocr.Reader(['en'])
    found = False
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        ocr_results = reader.readtext(crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        for _, text, conf in ocr_results:
            filtered = ''.join([c for c in text if c.isalnum()])
            if filtered:
                print(f"Detected: {filtered} (Confidence: {conf:.2f})")
                found = True
    if not found:
        print("No text/numbers detected in detected regions.")

if __name__ == "__main__":
    main()
