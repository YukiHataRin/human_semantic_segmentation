from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
RESULTS = ROOT / "results"
for directory in (UPLOADS, RESULTS):
    directory.mkdir(exist_ok=True)

app = FastAPI(title="Human Mask Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/results", StaticFiles(directory=RESULTS), name="results")
model = YOLO("yolo11n-seg.pt")
PERSON_CLASS = 0
# Compile/initialize CUDA before the first user upload, so the reported first result
# reflects inference instead of one-time model startup overhead.
model(np.zeros((640, 640, 3), dtype=np.uint8), classes=[PERSON_CLASS], verbose=False)

def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (245, 215, 32)
    return tuple(int(value[i : i + 2], 16) for i in (4, 2, 0))

def mask_frame(frame: np.ndarray, confidence: float, opacity: float, color: str) -> tuple[np.ndarray, int, float]:
    started = time.perf_counter()
    result = model(frame, classes=[PERSON_CLASS], conf=confidence, verbose=False)[0]
    output = frame.copy()
    people = 0
    if result.masks is not None:
        mask_color = np.array(hex_to_bgr(color), dtype=np.uint8)
        for mask in result.masks.data.cpu().numpy():
            resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
            output[resized] = (output[resized] * (1 - opacity) + mask_color * opacity).astype(np.uint8)
            people += 1
    return output, people, (time.perf_counter() - started) * 1000

async def save_upload(file: UploadFile) -> Path:
    extension = Path(file.filename or "media").suffix.lower() or ".bin"
    destination = UPLOADS / f"{uuid.uuid4().hex}{extension}"
    destination.write_bytes(await file.read())
    return destination

@app.post("/api/segment/image")
async def segment_image(file: UploadFile = File(...), confidence: float = Form(0.4), overlay_opacity: float = Form(0.55), overlay_color: str = Form("#20d7f5")):
    input_path = await save_upload(file)
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise HTTPException(400, "Please upload a readable image file.")
    output, people, elapsed = mask_frame(frame, confidence, overlay_opacity, overlay_color)
    output_name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(RESULTS / output_name), output)
    return {"result_url": f"/results/{output_name}", "metrics": {"inference_ms": elapsed, "people": people, "fps": 1000 / elapsed if elapsed else 0}}

@app.post("/api/segment/video")
async def segment_video(file: UploadFile = File(...), confidence: float = Form(0.4), overlay_opacity: float = Form(0.55), overlay_color: str = Form("#20d7f5")):
    input_path = await save_upload(file)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise HTTPException(400, "Please upload a readable video file.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_name = f"{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(str(RESULTS / output_name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    elapsed_total = 0.0; frame_count = 0; max_people = 0
    while True:
        ok, frame = capture.read()
        if not ok: break
        output, people, elapsed = mask_frame(frame, confidence, overlay_opacity, overlay_color)
        writer.write(output); elapsed_total += elapsed; frame_count += 1; max_people = max(max_people, people)
    capture.release(); writer.release()
    if not frame_count:
        raise HTTPException(400, "This video did not contain any frames.")
    average = elapsed_total / frame_count
    return {"result_url": f"/results/{output_name}", "metrics": {"inference_ms": average, "people": max_people, "fps": 1000 / average if average else 0}}
