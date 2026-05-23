from fastapi import UploadFile, HTTPException
from PIL import Image
import io
import numpy as np
import cv2
import math


async def read_image_from_upload(file: UploadFile) -> "np.ndarray":
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")
    arr = np.array(img)
    return arr


def resize_image(image: "np.ndarray", max_width: int = 1280) -> "np.ndarray":
    from PIL import Image
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    new_h = int(h * scale)
    pil = Image.fromarray(image)
    pil = pil.resize((max_width, new_h), Image.LANCZOS)
    return np.array(pil)


def crop_bbox(image: "np.ndarray", bbox: list) -> "np.ndarray":
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def align_face(face_img: "np.ndarray", output_size: int = 112) -> "np.ndarray":
    """Align a face image using eye detection.

    Returns an RGB image of shape (output_size, output_size, 3).
    If eyes aren't found, the function resizes the input to `output_size`.
    """
    # face_img expected RGB HxWx3
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    except Exception:
        return cv2.resize(face_img, (output_size, output_size))

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10))
    if len(eyes) < 2:
        # fallback: just resize
        return cv2.resize(face_img, (output_size, output_size))

    # compute eye centers
    centers = []
    for (ex, ey, ew, eh) in eyes:
        centers.append((ex + ew / 2.0, ey + eh / 2.0))
    # choose two eyes with largest horizontal separation
    centers = sorted(centers, key=lambda c: c[0])
    left = centers[0]
    right = centers[-1]

    dx = right[0] - left[0]
    dy = right[1] - left[1]
    angle = math.degrees(math.atan2(dy, dx))

    # rotate image to make eyes horizontal
    eyes_center = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    h, w = face_img.shape[:2]
    rotated = cv2.warpAffine(face_img, M, (w, h), flags=cv2.INTER_CUBIC)

    # crop square region centered at eyes_center
    side = int(max(h, w) * 0.9)
    cx, cy = int(eyes_center[0]), int(eyes_center[1])
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    cropped = rotated[y1:y2, x1:x2]
    if cropped.size == 0:
        return cv2.resize(face_img, (output_size, output_size))
    aligned = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return aligned
