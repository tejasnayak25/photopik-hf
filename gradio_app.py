import os
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import gradio as gr

from services import detection, embedding


def draw_boxes(image: np.ndarray, faces: list):
    pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f["bbox"]
        conf = f.get("confidence", 0)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = f"#{i} {conf:.2f}"
        if font:
            draw.text((x1 + 4, y1 + 4), label, fill=(255, 0, 0), font=font)
        else:
            draw.text((x1 + 4, y1 + 4), label, fill=(255, 0, 0))
    return pil


def process(image: np.ndarray):
    if image is None:
        return None, {}
    # ensure RGB numpy
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    faces = detection.detect_faces(image)
    results = []
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        crop = image[y1:y2, x1:x2]
        emb = embedding.embed_face(crop)
        results.append({"bbox": f["bbox"], "confidence": f.get("confidence", 0), "embedding": emb})
    out_img = draw_boxes(image, faces)
    return out_img, results


with gr.Blocks() as demo:
    gr.Markdown("# Photopik — Face Embedding Service")
    with gr.Row():
        inp = gr.Image(type="numpy", label="Upload image")
        out_img = gr.Image(type="pil", label="Detections")
    out_json = gr.JSON(label="Face results")
    inp.change(fn=process, inputs=inp, outputs=[out_img, out_json])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("HF_SPACE_PORT", 7860)))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
