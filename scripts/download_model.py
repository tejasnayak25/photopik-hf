#!/usr/bin/env python3
"""Download an ONNX embedding model into photopik-hf/models/embedding.onnx

Usage:
  python scripts/download_model.py <URL> [--commit]

If --commit is passed the script will also `git add` and `git commit` the downloaded file.
You still need to `git push` to your remote (example: `git push Huggingface HEAD:main`).
"""
import sys
import os
import hashlib
import requests
import tempfile
import zipfile
import shutil


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_model.py <URL> [--commit]")
        sys.exit(2)

    url = sys.argv[1]
    do_commit = "--commit" in sys.argv[2:]

    out_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embedding.onnx")

    print(f"Downloading model from: {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    is_zip = url.lower().endswith('.zip') or 'zip' in content_type

    if is_zip:
        # download to temp file then inspect
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmpf:
            tmpname = tmpf.name
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    tmpf.write(chunk)
        print(f"Downloaded zip to temp file: {tmpname}")
        # inspect zip for .onnx files
        with zipfile.ZipFile(tmpname, 'r') as z:
            onnx_files = [info for info in z.infolist() if info.filename.lower().endswith('.onnx')]
            if not onnx_files:
                print("No .onnx files found inside the zip archive.")
                os.remove(tmpname)
                sys.exit(1)
            # prefer mobilefacenet-like names
            candidate = None
            for info in onnx_files:
                name = os.path.basename(info.filename).lower()
                if 'mobile' in name or 'mbf' in name or 'mobilefacenet' in name or 'mfn' in name:
                    candidate = info
                    break
            if candidate is None:
                # fallback: pick largest .onnx file
                candidate = max(onnx_files, key=lambda i: i.file_size)
            print(f"Extracting {candidate.filename} from zip to {out_path}")
            with z.open(candidate) as source, open(out_path, 'wb') as target:
                shutil.copyfileobj(source, target)
        os.remove(tmpname)
        total = os.path.getsize(out_path)
        print(f"Saved to: {out_path} ({total} bytes)")
        print("Computing SHA256...")
        digest = sha256_file(out_path)
        print(f"SHA256: {digest}")
    else:
        # direct onnx download
        total = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        print(f"Saved to: {out_path} ({total} bytes)")
        print("Computing SHA256...")
        digest = sha256_file(out_path)
        print(f"SHA256: {digest}")

    if do_commit:
        import subprocess
        try:
            subprocess.check_call(["git", "add", out_path])
            subprocess.check_call(["git", "commit", "-m", "Add MobileFaceNet embedding.onnx model"]) 
            print("Committed model to git. Run `git push Huggingface HEAD:main` to deploy.")
        except Exception as e:
            print("Git commit failed:", e)


if __name__ == "__main__":
    main()
