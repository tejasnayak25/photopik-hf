import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from services import detection, embedding
from utils.image import read_image_from_upload
from fastapi import Request
import time
import threading
from collections import defaultdict, deque


def _get_allowed_origins():
    val = os.environ.get("ALLOWED_ORIGINS")
    if not val:
        return ["*"]
    return [o.strip() for o in val.split(",") if o.strip()]


def get_api_key(x_api_key: str | None = Header(None)):
    """Require `x-api-key` when `API_KEY` env var is set."""
    configured = os.environ.get("API_KEY")
    if not configured:
        return None
    if not x_api_key or x_api_key != configured:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key


app = FastAPI(title="Face Embedding Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple in-memory rate limiter (per-IP fixed window)
_RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
_RATE_WINDOW = 60  # seconds
_COUNTERS = defaultdict(lambda: deque())
_LOCK = threading.Lock()


def _client_ip(request: Request) -> str:
    # prefer X-Forwarded-For if present
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def rate_limiter(request: Request):
    if _RATE_LIMIT <= 0:
        return
    ip = _client_ip(request)
    now = time.time()
    window_start = now - _RATE_WINDOW
    with _LOCK:
        dq = _COUNTERS[ip]
        # prune old timestamps
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Too many requests")
        dq.append(now)



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), _api_key: str = Depends(get_api_key), _rl=Depends(rate_limiter)):
    image = await read_image_from_upload(file)
    faces = detection.detect_faces(image)
    return JSONResponse({"faces": faces})


@app.post("/embed")
async def embed(file: UploadFile = File(...), _api_key: str = Depends(get_api_key), _rl=Depends(rate_limiter)):
    image = await read_image_from_upload(file)
    emb = embedding.embed_face(image)
    if emb is None:
        raise HTTPException(status_code=500, detail="Embedding model not available or failed to produce embeddings")
    # detect suspicious all-zero vector
    try:
        if all([float(v) == 0.0 for v in emb]):
            raise HTTPException(status_code=500, detail="Embedding model produced all-zero vector — check model load")
    except Exception:
        pass
    return JSONResponse({"embedding": emb})


@app.post("/process")
async def process(file: UploadFile = File(...), _api_key: str = Depends(get_api_key), _rl=Depends(rate_limiter)):
    image = await read_image_from_upload(file)
    faces = detection.detect_faces(image)
    results = []
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        crop = image[y1:y2, x1:x2]
        emb = embedding.embed_face(crop)
        if emb is None:
            # surface issue but continue returning detections with null embedding
            print("WARNING: embed_face returned None for a detected face")
        else:
            try:
                if all([float(v) == 0.0 for v in emb]):
                    print("WARNING: embed_face returned all-zero vector for a detected face")
            except Exception:
                pass

        results.append({"bbox": f["bbox"], "confidence": f["confidence"], "embedding": emb})
    return JSONResponse({"faces": results})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, log_level="info")
