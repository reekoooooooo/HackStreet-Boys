import cv2
import numpy as np
import easyocr
import re

# ── Mock GIS Database ─────────────────────────────────────────────────────────
GIS_DATABASE = {
    "444194":  {"type": "Wood Pole", "location": "123 Main St, Richmond VA",  "status": "Active"},
    "625296":  {"type": "Wood Pole", "location": "456 Elm St, Richmond VA",   "status": "Active"},
    "PD41459": {"type": "Pad Mount", "location": "789 Oak Ave, Richmond VA",  "status": "Active"},
    "D-8176":  {"type": "Metal Pad", "location": "321 Pine Rd, Richmond VA",  "status": "Active"},
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img):
    # Resize if too large
    h, w = img.shape[:2]
    if w > 1024 or h > 1024:
        scale = min(1024/w, 1024/h)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Sharpen
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return img

def check_quality(img):
    warnings = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
        warnings.append("Image may be too blurry.")
    brightness = np.mean(gray)
    if brightness < 50:
        warnings.append("Image is too dark.")
    elif brightness > 220:
        warnings.append("Image is overexposed.")
    return warnings

def extract_tag_number(ocr_results):
    for (_, text, confidence) in ocr_results:
        cleaned = text.strip().replace(" ", "")
        if confidence > 0.4 and re.match(r'^[A-Z0-9\-]{4,10}$', cleaned):
            return cleaned
    return None

# ── Main ──────────────────────────────────────────────────────────────────────
def process_image(image_path):
    print(f"\nProcessing: {image_path}")
    print("-" * 40)

    # Load
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Could not load image. Check the file path.")
        return

    # Quality check
    warnings = check_quality(img)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    else:
        print("Quality check: PASSED")

    # Preprocess
    processed = preprocess(img)
    cv2.imwrite("processed_output.jpg", processed)
    print("Preprocessed image saved as: processed_output.jpg")

    # OCR
    print("Reading tag number...")
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(processed)

    print("\nAll text found in image:")
    for (_, text, conf) in results:
        print(f"  '{text}' — confidence: {conf:.0%}")

    # Extract tag
    tag = extract_tag_number(results)

    print()
    if tag:
        print(f"Asset Tag Detected: {tag}")

        record = GIS_DATABASE.get(tag)
        if record:
            print("Database match FOUND:")
            print(f"  Type     : {record['type']}")
            print(f"  Location : {record['location']}")
            print(f"  Status   : {record['status']}")
        else:
            print("Tag NOT found in database — flag for manual review.")
    else:
        print("Could not extract a tag number. Check image quality and retry.")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process.py <image_path>")
        print("Example: python process.py PoleTag_25.jpg")
    else:
        process_image(sys.argv[1])