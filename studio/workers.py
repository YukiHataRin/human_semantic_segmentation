from __future__ import annotations

import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from .engine import ROOT, SegmentationEngine, Settings, encode_video
from .cameras import Camera


class LatestFrame:
    """A single-frame mailbox; capture never queues stale camera frames."""

    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.closed = False

    def put(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify()

    def take(self, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.frame is not None or self.closed, timeout)
            frame, self.frame = self.frame, None
            return frame

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def open_capture(source, camera=False):
    if isinstance(source, Camera):
        capture = cv2.VideoCapture(source.index, source.backend)
    elif camera and sys.platform == "darwin":
        capture = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    else:
        capture = cv2.VideoCapture(source)
    if camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class CameraReader(threading.Thread):
    def __init__(self, capture, startup_timeout=5):
        super().__init__(daemon=True, name="camera-capture")
        self.capture = capture
        self.frames = LatestFrame()
        self.stop_event = threading.Event()
        self.error = None
        self.startup_timeout = startup_timeout

    def run(self):
        deadline = time.monotonic() + self.startup_timeout
        received = False
        try:
            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    if not received and time.monotonic() < deadline:
                        self.stop_event.wait(0.05)
                        continue
                    if not self.stop_event.is_set():
                        self.error = "The camera stopped delivering frames. Check its connection and camera permission."
                    break
                received = True
                self.frames.put(frame)
        except Exception as error:
            self.error = str(error)
        finally:
            self.capture.release()
            self.frames.close()

    def stop(self):
        self.stop_event.set()
        self.frames.close()


class ProcessingWorker(QThread):
    status = Signal(str)
    device_ready = Signal(str)
    failed = Signal(str)
    completed = Signal(object)

    def __init__(self, mode, source, settings, parent=None, engine_factory=SegmentationEngine, capture_factory=open_capture):
        super().__init__(parent)
        self.mode, self.source = mode, source
        self._settings = settings
        self._lock = threading.Lock()
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.preview = None
        self.engine_factory = engine_factory
        self.capture_factory = capture_factory

    def update_settings(self, settings):
        with self._lock:
            self._settings = settings

    def settings(self):
        with self._lock:
            return self._settings

    def take_preview(self):
        with self._lock:
            preview, self.preview = self.preview, None
            return preview

    def stop(self):
        self.stop_event.set()

    def publish(self, result, processed, total, tracks, started):
        with self._lock:
            self.preview = (result, {"frames": processed, "total": total, "tracks": len(tracks), "fps": processed / max(0.001, time.perf_counter() - started)})

    def run(self):
        capture = writer = reader = None
        working_path = None
        try:
            self.status.emit("Loading YOLO11n…")
            engine = self.engine_factory(self.settings().device)
            self.status.emit(f"Warming up {engine.device.label}…")
            engine.warmup()
            engine.reset()
            self.device_ready.emit(engine.device.label)
            if self.stop_event.is_set():
                self.completed.emit({"stopped": True, "path": None})
                return
            if self.mode == "image":
                frame = cv2.imread(str(self.source))
                if frame is None:
                    raise RuntimeError("Could not read this image. Choose a supported image file.")
                started = time.perf_counter()
                result = engine.process(frame, self.settings())
                self.publish(result, 1, 1, set(), started)
                self.completed.emit({"stopped": self.stop_event.is_set(), "path": None})
                return

            self.status.emit("Opening camera…" if self.mode == "camera" else "Opening video…")
            capture = self.capture_factory(self.source, camera=self.mode == "camera")
            if not capture.isOpened():
                raise RuntimeError("Could not open the camera. Check its connection and camera permission, then refresh cameras." if self.mode == "camera" else "Could not read this video. Choose a supported video file.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if 0 < fps <= 240 else 30
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.mode == "video" else 0
            processed = 0
            tracks = set()
            started = time.perf_counter()
            tracking = self.settings().tracking
            output_path = None
            if self.mode == "camera":
                reader = CameraReader(capture)
                reader.start()
            self.status.emit("Waiting for camera frames…" if self.mode == "camera" else "Processing video")
            while not self.stop_event.is_set():
                if self.paused.is_set():
                    self.stop_event.wait(0.05)
                    continue
                tick = time.perf_counter()
                if reader:
                    frame = reader.frames.take()
                    if frame is None:
                        if reader.frames.closed:
                            raise RuntimeError(reader.error or "Camera disconnected.")
                        continue
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                settings = self.settings()
                if settings.tracking != tracking:
                    engine.reset()
                    tracks.clear()
                    tracking = settings.tracking
                result = engine.process(frame, settings, tracking=tracking)
                if self.mode == "camera" and processed == 0:
                    self.status.emit("Camera live")
                tracks.update(result.track_ids)
                processed += 1
                if self.mode == "video":
                    if writer is None:
                        output_dir = ROOT / "outputs"
                        output_dir.mkdir(exist_ok=True)
                        output_path = output_dir / f"{Path(self.source).stem}-{uuid.uuid4().hex[:8]}.mp4"
                        working_path = output_path.with_suffix(".working.mp4")
                        height, width = result.image.shape[:2]
                        width, height = width - width % 2, height - height % 2
                        writer = cv2.VideoWriter(str(working_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("Could not initialize the video encoder.")
                    writer.write(result.image[:height, :width])
                self.publish(result, processed, total, tracks, started)
                if self.mode == "video":
                    self.stop_event.wait(max(0, 1 / fps - (time.perf_counter() - tick)))
            if writer:
                writer.release()
                writer = None
            if not processed and not self.stop_event.is_set():
                raise RuntimeError("This source did not contain any readable frames.")
            if output_path and not self.stop_event.is_set():
                self.status.emit("Saving MP4…")
                encode_video(working_path, output_path)
            self.completed.emit({"stopped": self.stop_event.is_set(), "path": output_path if not self.stop_event.is_set() else None})
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if reader:
                reader.stop()
                reader.join()
            elif capture:
                capture.release()
            if writer:
                writer.release()
            if working_path:
                working_path.unlink(missing_ok=True)
