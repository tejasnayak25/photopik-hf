import os
import numpy as np
import cv2
from utils.image import align_face
import traceback
from datetime import datetime
import multiprocessing

try:
    import onnxruntime as ort
except Exception:
    ort = None


EMB_DIM = 512


def _model_path():
    return os.path.join(os.getcwd(), "models", "embedding.onnx")


def write_log(msg: str):
    """Append a timestamped message to a local debug log file and print it."""
    try:
        timestamp = datetime.utcnow().isoformat() + "Z"
        line = f"[{timestamp}] {msg}\n"
        # try writing to /tmp first, then cwd
        for p in ("/tmp/photopik_embed.log", os.path.join(os.getcwd(), "photopik_embed.log")):
            try:
                with open(p, "a", encoding="utf8") as f:
                    f.write(line)
                break
            except Exception:
                continue
        print(line.strip())
    except Exception:
        try:
            print("Failed to write debug log", msg)
        except Exception:
            pass


def load_embedding_model():
    path = _model_path()
    try:
        if ort is None:
            write_log("onnxruntime (ort) is not installed or failed to import")
            return None
        if not os.path.exists(path):
            write_log(f"Embedding model not found at: {path}")
            return None
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = max(1, multiprocessing.cpu_count())
        so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", str(max(1, cpu_count // 2))))
        so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        write_log(f"Loaded embedding model from {path}")
        return sess
    except Exception as e:
        write_log(f"Exception loading embedding model: {e}\n" + traceback.format_exc())
        return None


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
        write_log("embed_face called but embedding model session is None")
        return None

    try:
        align_flag = os.environ.get("ALIGN_FACES", "0").lower() in ("1", "true", "yes")
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
    except Exception as e:
        write_log(f"Exception during embed_face: {e}\n" + traceback.format_exc())
        return None


