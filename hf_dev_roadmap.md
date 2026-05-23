# Hugging Face Space — Development Roadmap
## Face Embedding AI Service

This roadmap focuses on building the AI backend for a face-based photo retrieval system using Hugging Face Spaces.

---

## Goal

Create a lightweight AI service that can:

- detect faces in an uploaded image
- crop each detected face
- generate face embeddings
- return structured JSON for matching later

---

## What the HF Space Should Do

The Space should only handle AI processing:

- image upload
- face detection
- face cropping
- embedding generation
- JSON response

It should **not**:

- store images permanently
- manage users
- manage gallery UI
- handle event logic
- perform authentication
- do long-term search indexing

---

## Recommended Stack

- **Space type:** Docker Space
- **API framework:** FastAPI
- **Face detection model:** SCRFD
- **Face embedding model:** ArcFace / InsightFace
- **Runtime:** ONNX Runtime CPU
- **Image handling:** OpenCV + Pillow
- **Output format:** JSON

---

## Architecture

```txt
Frontend (Vercel)
    ↓
Next.js API Routes
    ↓
Hugging Face Space
    ↓
Face detection + embeddings
    ↓
Return JSON
```

---

# Phase 0 — Planning

## Objectives

Define:

- input format
- output format
- model choice
- performance target
- error handling strategy

## Decisions to lock early

- embedding dimension: 512D
- image size limit
- accepted file types
- similarity threshold range
- whether the endpoint accepts full images or face crops

---

# Phase 1 — Create the Space

## Tasks

- create a new Hugging Face Space
- choose Docker Space for flexibility
- set up a FastAPI app
- add dependencies
- make a health check route

## Minimum files

```txt
/app.py
/requirements.txt
/Dockerfile
```

## Basic requirements

```txt
fastapi
uvicorn
numpy
pillow
opencv-python-headless
onnxruntime
insightface
python-multipart
```

## Deliverables

- working public Space
- API starts successfully
- health endpoint returns OK

---

# Phase 2 — Load Models

## Objectives

Make model loading stable and efficient.

## Tasks

- download and store model files in the Space
- load models once at startup
- avoid reloading per request
- test startup time and memory usage

## Recommended model setup

- detection: SCRFD
- recognition: ArcFace
- runtime: ONNX Runtime CPU

## Important rules

- do not load large PyTorch models if ONNX is enough
- do not fetch weights on every request
- keep model files versioned and consistent

## Deliverables

- models load on startup
- inference works locally in the Space
- startup time is acceptable

---

# Phase 3 — Face Detection Endpoint

## Endpoint

```txt
POST /detect
```

## Input

- image file upload

## Output

```json
{
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.98
    }
  ]
}
```

## Tasks

- validate image input
- resize large images before inference
- detect one or more faces
- return bounding boxes and confidence values

## Deliverables

- multiple face detection works
- empty-face case handled cleanly
- response format is stable

---

# Phase 4 — Embedding Endpoint

## Endpoint

```txt
POST /embed
```

## Input

- cropped face image
- or full image plus face box, depending on your design

## Output

```json
{
  "embedding": [0.12, -0.44, ...]
}
```

## Tasks

- align face if needed
- normalize image before inference
- generate embedding vector
- return consistent numeric output

## Recommendations

- keep embeddings in float32 during inference
- compress to float16 only when storing elsewhere
- ensure the same preprocessing is used every time

## Deliverables

- embedding generation works
- output vector is stable
- model output is reproducible

---

# Phase 5 — Full Pipeline Endpoint

## Endpoint

```txt
POST /process
```

## Purpose

Handle the complete flow in one request.

## Pipeline

```txt
Image
→ detect faces
→ crop faces
→ generate embeddings
→ return all face results
```

## Output

```json
{
  "faces": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.98,
      "embedding": [ ... ]
    }
  ]
}
```

## Deliverables

- one endpoint for production use
- multi-face support
- stable JSON response for the frontend

---

# Phase 6 — Performance Optimization

## Objectives

Keep the Space usable on CPU.

## Tasks

- resize input images before detection
- reject very large uploads
- reuse model objects in memory
- minimize image copies
- keep processing sequential for free-tier usage

## Recommended limits

- max upload size: around 10 MB
- max working width: around 1280 px
- crop and process faces instead of full-resolution images when possible

## Deliverables

- faster inference
- lower RAM usage
- fewer timeout issues

---

# Phase 7 — Error Handling

## Objectives

Make the API predictable.

## Tasks

- validate file type
- validate file size
- handle invalid image data
- handle no-face detection cleanly
- return useful error messages

## Suggested error cases

- unsupported format
- corrupted image
- no face detected
- inference timeout
- model load failure

## Deliverables

- clean error responses
- stable behavior under bad input
- easier frontend integration

---

# Phase 8 — Security

## Objectives

Prevent abuse.

## Tasks

- add API key protection
- restrict CORS to your frontend domain
- add rate limiting if possible
- avoid public write access
- never expose model internals

## Important note

The Space will process biometric data, so keep access controlled and minimal.

## Deliverables

- protected API routes
- limited public exposure
- safer production usage

---

# Phase 9 — Integration with Main App

## Objectives

Connect the HF Space to the main application.

## Main app flow

```txt
User uploads selfie
→ Next.js API route
→ HF Space /process
→ embeddings returned
→ Firestore stores metadata
→ matching results shown in UI
```

## Tasks

- define request payload format
- define response format
- add retry handling
- test end-to-end upload and retrieval

## Deliverables

- frontend successfully calls Space
- face embeddings stored in database
- search pipeline works end-to-end

---

# Phase 10 — Production Hardening

## Objectives

Make the Space reliable.

## Tasks

- benchmark inference time
- monitor memory usage
- test cold starts
- add warmup ping behavior
- log failures and slow requests
- version models carefully

## Deliverables

- predictable startup behavior
- acceptable CPU performance
- fewer failures under load

---

# Suggested Project Structure

```txt
/app.py
/models
/services
/utils
/requirements.txt
/Dockerfile
```

## Example responsibilities

- `app.py`: FastAPI routes
- `services/detection.py`: face detection logic
- `services/embedding.py`: embedding generation logic
- `utils/image.py`: resize, crop, normalize helpers

---

# Recommended Build Order

```txt
1. Create Space
2. Load models
3. Build /detect
4. Build /embed
5. Build /process
6. Optimize performance
7. Secure API
8. Connect to frontend
```

---

# MVP Definition

The HF Space MVP is complete when:

- face detection works
- embeddings are generated
- `/process` returns JSON for multiple faces
- the Space runs on CPU reliably
- the frontend can call it successfully

---

# Future Upgrades

Later, you can move to:

- GPU Space for faster inference
- async queues for batch uploads
- better vector storage
- face clustering
- video frame indexing
