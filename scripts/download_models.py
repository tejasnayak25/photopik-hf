"""Download model artifacts into the local `models/` folder.

Usage:
  python scripts/download_models.py

You can also set environment variable `MODEL_URLS` with a JSON mapping
of filename -> url to override the defaults.

Defaults download:
- deploy.prototxt and res10_300x300.caffemodel (OpenCV DNN face detector)
"""
import os
import json
import sys
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


DEFAULTS = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300.caffemodel",
    # Add your embedding ONNX under the name `embedding.onnx` if available
    # "embedding.onnx": "https://example.com/path/to/embedding.onnx",
}


def load_model_map():
    env = os.environ.get("MODEL_URLS")
    if env:
        try:
            return json.loads(env)
        except Exception as e:
            print("Failed to parse MODEL_URLS:", e)
    # fallback to defaults
    return DEFAULTS


def ensure_models_dir(path="models"):
    os.makedirs(path, exist_ok=True)
    return path


def download(url, dest_path):
    if os.path.exists(dest_path):
        print(f"Skip (exists): {dest_path}")
        return
    print(f"Downloading {url} → {dest_path}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def main():
    mapping = load_model_map()
    models_dir = ensure_models_dir()
    for name, url in mapping.items():
        # basic validation
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            print("Skipping invalid URL:", url)
            continue
        dest = os.path.join(models_dir, name)
        try:
            download(url, dest)
        except Exception as e:
            print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    main()
