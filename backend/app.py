from __future__ import annotations

import time
import uuid
import subprocess
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
TRACKER_CONFIG = ROOT / "botsort_reid.yaml"
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
TRACK_COLORS = (
    "#22D3EE",
    "#FB7185",
    "#A78BFA",
    "#FBBF24",
    "#34D399",
    "#F97316",
    "#60A5FA",
    "#E879F9",
)
# Compile/initialize CUDA before the first user upload, so the reported first result
# reflects inference instead of one-time model startup overhead.
model(np.zeros((640, 640, 3), dtype=np.uint8), classes=[PERSON_CLASS], verbose=False)

def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (245, 215, 32)
    return tuple(int(value[i : i + 2], 16) for i in (4, 2, 0))

def color_for_track(track_id: int) -> tuple[int, int, int]:
    return hex_to_bgr(TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)])

def reset_tracker() -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", []):
        tracker.reset()

def encode_browser_video(source: Path, destination: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        destination.unlink(missing_ok=True)
        detail = error.stderr.decode(errors="replace") if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise HTTPException(500, f"Could not encode a browser-compatible video: {detail}") from error
    finally:
        source.unlink(missing_ok=True)

def compose_masks(
    frame: np.ndarray,
    result,
    opacity: float,
    color: str,
    output_mode: str,
    track_ids: list[int] | None = None,
) -> tuple[np.ndarray, int]:
    output = np.zeros_like(frame) if output_mode == "mask" else frame.copy()
    if result.masks is None:
        return output, 0

    fallback_color = np.array(hex_to_bgr(color), dtype=np.uint8)
    boxes = result.boxes.xyxy.int().cpu().numpy() if result.boxes is not None else []
    for index, mask in enumerate(result.masks.data.cpu().numpy()):
        resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
        track_id = track_ids[index] if track_ids and index < len(track_ids) else None
        mask_color = np.array(color_for_track(track_id), dtype=np.uint8) if track_id is not None else fallback_color
        if output_mode == "mask" and track_id is None:
            output[resized] = (255, 255, 255)
        elif output_mode == "mask":
            output[resized] = mask_color
        else:
            output[resized] = (output[resized] * (1 - opacity) + mask_color * opacity).astype(np.uint8)

        if track_id is not None and index < len(boxes):
            x1, y1, _, _ = boxes[index]
            label = f"ID {track_id}"
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            label_height = text_size[1] + baseline + 10
            label_left = max(0, min(x1, frame.shape[1] - text_size[0] - 14))
            if y1 >= label_height:
                label_top, label_bottom = y1 - label_height, y1
                text_y = y1 - baseline - 4
            else:
                label_top, label_bottom = y1, min(frame.shape[0] - 1, y1 + label_height)
                text_y = min(frame.shape[0] - baseline - 1, y1 + text_size[1] + 4)
            cv2.rectangle(output, (label_left, label_top), (label_left + text_size[0] + 14, label_bottom), color_for_track(track_id), -1)
            cv2.putText(output, label, (label_left + 7, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 13, 15), 2, cv2.LINE_AA)
    return output, len(result.masks.data)

def mask_frame(frame: np.ndarray, confidence: float, opacity: float, color: str, output_mode: str = "overlay") -> tuple[np.ndarray, int, float]:
    started = time.perf_counter()
    result = model(frame, classes=[PERSON_CLASS], conf=confidence, verbose=False)[0]
    output, people = compose_masks(frame, result, opacity, color, output_mode)
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
async def segment_video(file: UploadFile = File(...), confidence: float = Form(0.4), overlay_opacity: float = Form(0.55), overlay_color: str = Form("#20d7f5"), output_mode: str = Form("overlay"), enable_tracking: bool = Form(True)):
    input_path = await save_upload(file)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise HTTPException(400, "Please upload a readable video file.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_name = f"{uuid.uuid4().hex}.mp4"
    working_path = RESULTS / f"{uuid.uuid4().hex}.working.mp4"
    writer = cv2.VideoWriter(str(working_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise HTTPException(500, "Could not initialize the video encoder.")
    elapsed_total = 0.0; frame_count = 0; max_people = 0
    unique_track_ids: set[int] = set()
    if enable_tracking:
        reset_tracker()
    while True:
        ok, frame = capture.read()
        if not ok: break
        if enable_tracking:
            started = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker=str(TRACKER_CONFIG),
                classes=[PERSON_CLASS],
                conf=confidence,
                verbose=False,
            )[0]
            track_ids = [] if result.boxes.id is None else result.boxes.id.int().cpu().tolist()
            unique_track_ids.update(track_ids)
            output, people = compose_masks(frame, result, overlay_opacity, overlay_color, output_mode, track_ids)
            elapsed = (time.perf_counter() - started) * 1000
        else:
            output, people, elapsed = mask_frame(frame, confidence, overlay_opacity, overlay_color, output_mode)
        writer.write(output); elapsed_total += elapsed; frame_count += 1; max_people = max(max_people, people)
    capture.release(); writer.release()
    if not frame_count:
        working_path.unlink(missing_ok=True)
        raise HTTPException(400, "This video did not contain any frames.")
    encode_browser_video(working_path, RESULTS / output_name)
    average = elapsed_total / frame_count
    return {
        "result_url": f"/results/{output_name}",
        "metrics": {
            "inference_ms": average,
            "people": max_people,
            "unique_tracks": len(unique_track_ids),
            "fps": 1000 / average if average else 0,
        },
    }
