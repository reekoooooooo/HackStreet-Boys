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

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("ERROR: Could not load image.")
        sys.exit(1)

    output = preprocess(img)
    cv2.imwrite("processed_output.jpg", output)
    print("Done. Saved as processed_output.jpg")