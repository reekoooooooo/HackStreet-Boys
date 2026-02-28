from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
import re

app = Flask(__name__)
CORS(app)

# Load OCR once at startup
print("Loading OCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("Ready.")

# ── Copy your preprocess functions here ──────────────────────────────────────

def preprocess_original(img):
    return img

def preprocess_standard(img):
    h, w = img.shape[:2]
    if w > 1024 or h > 1024:
        scale = min(1024/w, 1024/h)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img

def preprocess_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preprocess_inverted(img):
    return cv2.bitwise_not(preprocess_threshold(img))

def try_all_preprocesses(img):
    candidates = []
    for version in [
        preprocess_original(img),
        preprocess_standard(img),
        preprocess_threshold(img),
        preprocess_inverted(img)
    ]:
        results = reader.readtext(version)
        candidates.extend(results)
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates

def extract_tag_number(ocr_results):
    for (_, text, conf) in ocr_results:
        cleaned = text.strip().replace(" ", "")
        if conf > 0.6 and re.match(r'^[A-Z0-9\-]{4,10}$', cleaned):
            return cleaned, "high_confidence"
        elif conf > 0.4 and re.match(r'^[A-Z0-9\-]{4,10}$', cleaned):
            return cleaned, "low_confidence"
    return None, "unreadable"

# ── Route ─────────────────────────────────────────────────────────────────────

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    results = try_all_preprocesses(img)
    tag, confidence = extract_tag_number(results)

    return jsonify({"tag": tag, "confidence": confidence})

if __name__ == '__main__':
    app.run(port=5000, debug=True)