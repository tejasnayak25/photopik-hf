---
title: Face Embedding Service
emoji: "📷"
colorFrom: gray
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Face Embedding Service (HF Space)

Minimal scaffold for a Hugging Face Docker Space that detects faces and returns embeddings.

Quick start (local):

```bash
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Endpoints:
- `GET /health` — health check
- `POST /detect` — upload image -> returns bounding boxes
- `POST /embed` — upload face image -> returns embedding
- `POST /process` — full pipeline: detect -> crop -> embed

Next steps: implement model loading in `services/` and add tests.

Configuration (environment variables):
- `API_KEY` — if set, requests must include header `x-api-key: <API_KEY>`.
- `ALLOWED_ORIGINS` — comma-separated list of allowed CORS origins (defaults to `*`).

- `RATE_LIMIT_PER_MIN` — max requests per minute per IP (defaults to `60`).

- `ALIGN_FACES` — whether to perform eye-based alignment before embedding (defaults to `1`).

Run with an API key locally:

```bash
export API_KEY=your_secret_key
uvicorn app:app --host 0.0.0.0 --port 8080
```

Downloading models
------------------

The repository supports an automated download script for necessary model files.
By default it downloads the OpenCV DNN face detector into `models/`.

```bash
python scripts/download_models.py
```

To supply custom model URLs, set `MODEL_URLS` to a JSON mapping (filename→url), for example:

```bash
export MODEL_URLS='{"embedding.onnx":"https://.../embedding.onnx"}'
python scripts/download_models.py
```


