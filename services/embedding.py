import os
import numpy as np
import cv2
from utils.image import align_face

try:
    import onnxruntime as ort
except Exception:
    ort = None


EMB_DIM = 512


def _model_path():
    return os.path.join(os.getcwd(), "models", "embedding.onnx")


def load_embedding_model():
    path = _model_path()
    if ort is None or not os.path.exists(path):
        return None
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess


_SESSION = load_embedding_model()


def _preprocess(img: "np.ndarray", size: int = 112):
    # img: HxWx3 RGB
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32)
    # normalization common for ArcFace: (x - 127.5) / 128.0
    img = (img - 127.5) / 128.0
    # transpose to CHW
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    return img


def embed_face(image: "np.ndarray"):
    """Produce an embedding for a face image.

    If an ONNX model exists at `./models/embedding.onnx` it will be used.
    Otherwise returns a zero vector of `EMB_DIM`.
    """
    global EMB_DIM
    if _SESSION is None:
        # No model available — return None so callers can detect and warn.
        # Returning a zero vector silently hides configuration/load problems.
        return None

    try:
        align_flag = os.environ.get("ALIGN_FACES", "1").lower() in ("1", "true", "yes")
        if align_flag:
            try:
                image = align_face(image, output_size=112)
            except Exception:
                pass
        inp = _preprocess(image)
        input_name = _SESSION.get_inputs()[0].name
        outputs = _SESSION.run(None, {input_name: inp})
        vec = outputs[0].ravel().astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        EMB_DIM = vec.shape[0]
        return vec.tolist()
    except Exception:
        # On error, surface failure so callers can handle it explicitly.
        return None

