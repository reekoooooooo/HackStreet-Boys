import cv2
import numpy as np
import sys

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    import os
    path = sys.argv[1]
    print(f"Loading image from: {path}")

    img = cv2.imread(path)
    if img is None:
        print("ERROR: Could not load image. Check the path is correct.")
        sys.exit(1)

    print("Image loaded successfully.")
    output = preprocess(img)

    # Generate output filename based on input and save to 'processed' folder
    processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    base, ext = os.path.splitext(os.path.basename(path))
    output_filename = f"{base}_processed{ext}"
    output_path = os.path.join(processed_dir, output_filename)
    cv2.imwrite(output_path, output)
    print(f"Done. Saved as {output_path}")