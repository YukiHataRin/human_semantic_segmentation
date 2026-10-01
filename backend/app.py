from __future__ import annotations

import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from studio.engine import SegmentationEngine, Settings, encode_video

ROOT = Path(__file__).resolve().parent
UPLOADS, RESULTS = ROOT / 'uploads', ROOT / 'results'
for directory in (UPLOADS, RESULTS):
    directory.mkdir(exist_ok=True)
engine = None
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global engine
    engine = await run_in_threadpool(SegmentationEngine)
    await run_in_threadpool(engine.warmup)
    yield


app = FastAPI(title='Human Mask Studio API', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=r'^http://(localhost|127\.0\.0\.1):\d+$', allow_methods=['*'], allow_headers=['*'])
app.mount('/results', StaticFiles(directory=RESULTS), name='results')


def save_upload(file):
    destination = UPLOADS / f"{uuid.uuid4().hex}{Path(file.filename or 'media').suffix.lower() or '.bin'}"
    with destination.open('wb') as output:
        shutil.copyfileobj(file.file, output)
    return destination


@app.get('/api/device')
def device_info():
    return {'device': engine.device.key, 'label': engine.device.label}


@app.post('/api/segment/image')
def segment_image(file: UploadFile = File(...), confidence: float = Form(0.4), overlay_opacity: float = Form(0.55), overlay_color: str = Form('#20d7f5')):
    input_path = save_upload(file)
    try:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise HTTPException(400, 'Please upload a readable image file.')
        with inference_lock:
            result = engine.process(frame, Settings(confidence=confidence, opacity=overlay_opacity, color=overlay_color))
        output_name = f'{uuid.uuid4().hex}.jpg'
        if not cv2.imwrite(str(RESULTS / output_name), result.image):
            raise HTTPException(500, 'Could not save the result image.')
        return {'result_url': f'/results/{output_name}', 'metrics': {'inference_ms': result.inference_ms, 'people': result.people, 'fps': 1000 / result.inference_ms if result.inference_ms else 0}}
    finally:
        input_path.unlink(missing_ok=True)


@app.post('/api/segment/video')
def segment_video(file: UploadFile = File(...), confidence: float = Form(0.4), overlay_opacity: float = Form(0.55), overlay_color: str = Form('#20d7f5'), output_mode: str = Form('overlay'), enable_tracking: bool = Form(True)):
    input_path = save_upload(file)
    working_path = RESULTS / f'{uuid.uuid4().hex}.working.mp4'
    output_name = f'{uuid.uuid4().hex}.mp4'
    capture = cv2.VideoCapture(str(input_path))
    writer = None
    try:
        if not capture.isOpened():
            raise HTTPException(400, 'Please upload a readable video file.')
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width, height = width - width % 2, height - height % 2
        writer = cv2.VideoWriter(str(working_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise HTTPException(500, 'Could not initialize the video encoder.')
        settings = Settings(confidence=confidence, opacity=overlay_opacity, color=overlay_color, output_mode=output_mode)
        elapsed_total = 0
        frame_count = max_people = 0
        tracks = set()
        with inference_lock:
            engine.reset()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = engine.process(frame, settings, tracking=enable_tracking)
                writer.write(result.image[:height, :width])
                elapsed_total += result.inference_ms
                frame_count += 1
                max_people = max(max_people, result.people)
                tracks.update(result.track_ids)
        writer.release()
        writer = None
        if not frame_count:
            raise HTTPException(400, 'This video did not contain any frames.')
        encode_video(working_path, RESULTS / output_name)
        average = elapsed_total / frame_count
        return {'result_url': f'/results/{output_name}', 'metrics': {'inference_ms': average, 'people': max_people, 'unique_tracks': len(tracks), 'fps': 1000 / average if average else 0}}
    except RuntimeError as error:
        raise HTTPException(500, str(error)) from error
    finally:
        capture.release()
        if writer:
            writer.release()
        input_path.unlink(missing_ok=True)
        working_path.unlink(missing_ok=True)
