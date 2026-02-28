import sys
import os
import cv2
import easyocr
from preprocess import preprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_reader.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading image from: {path}")
    img = cv2.imread(path)
    if img is None:
        print("ERROR: Could not load image. Check the path is correct.")
        sys.exit(1)

    print("Preprocessing image...")
    processed_img = preprocess(img)
    base, ext = os.path.splitext(os.path.basename(path))
    processed_filename = f"{base}_processed{ext}"
    cv2.imwrite(processed_filename, processed_img)
    print(f"Processed image saved as {processed_filename}")

    print("Running OCR...")
    reader = easyocr.Reader(['en'])
    results = reader.readtext(processed_filename)

    print("\nOCR Results:")
    for bbox, text, conf in results:
        filtered = ''.join([c for c in text if c.isalnum()])
        if filtered:
            print(f"Detected: {filtered} (Confidence: {conf:.2f})")

if __name__ == "__main__":
    main()
