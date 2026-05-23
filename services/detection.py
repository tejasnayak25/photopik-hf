import os
import numpy as np
import cv2
from utils.image import resize_image


def _dnn_paths():
    # Allow model files in ./models/ for optional faster detector
    base = os.path.join(os.getcwd(), "models")
    prototxt = os.path.join(base, "deploy.prototxt")
    caffemodel = os.path.join(base, "res10_300x300.caffemodel")
    if os.path.exists(prototxt) and os.path.exists(caffemodel):
        return prototxt, caffemodel
    return None, None


def detect_faces(image: np.ndarray, max_width: int = 1280):
    """Detect faces using OpenCV.

    Attempts a DNN SSD detector if model files are present at `./models/`.
    Falls back to Haar cascade provided by OpenCV.

    Returns list of dicts: {"bbox": [x1,y1,x2,y2], "confidence": float}
    """
    orig_h, orig_w = image.shape[:2]
    img = resize_image(image, max_width=max_width)
    h, w = img.shape[:2]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    prototxt, caffemodel = _dnn_paths()
    results = []
    if prototxt and caffemodel:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        blob = cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.3:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            # scale bbox back to original image coordinates
            x_scale = orig_w / float(w)
            y_scale = orig_h / float(h)
            sx1 = int(max(0, round(x1 * x_scale)))
            sy1 = int(max(0, round(y1 * y_scale)))
            sx2 = int(min(orig_w - 1, round(x2 * x_scale)))
            sy2 = int(min(orig_h - 1, round(y2 * y_scale)))
            results.append({"bbox": [sx1, sy1, sx2, sy2], "confidence": conf})
        return results

    # Fallback: Haar cascade
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        return []
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    x_scale = orig_w / float(w)
    y_scale = orig_h / float(h)
    for (x, y, wbox, hbox) in rects:
        x1, y1, x2, y2 = int(x), int(y), int(x + wbox), int(y + hbox)
        sx1 = int(max(0, round(x1 * x_scale)))
        sy1 = int(max(0, round(y1 * y_scale)))
        sx2 = int(min(orig_w - 1, round(x2 * x_scale)))
        sy2 = int(min(orig_h - 1, round(y2 * y_scale)))
        results.append({"bbox": [sx1, sy1, sx2, sy2], "confidence": 1.0})
    return results
