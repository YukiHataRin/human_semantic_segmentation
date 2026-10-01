from __future__ import annotations

import os
import platform
import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for config_directory in (os.environ["YOLO_CONFIG_DIR"], os.environ["MPLCONFIGDIR"]):
    Path(config_directory).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

TRACKER_CONFIG = ROOT / "backend" / "botsort_reid.yaml"
TRACK_COLORS = ("#22D3EE", "#FB7185", "#A78BFA", "#FBBF24", "#34D399", "#F97316", "#60A5FA", "#E879F9")


@dataclass(frozen=True)
class Device:
    key: str
    label: str


def available_devices() -> list[Device]:
    devices = []
    if torch.cuda.is_available():
        devices.extend(Device(f"cuda:{index}", f"CUDA · {torch.cuda.get_device_name(index)}") for index in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        devices.append(Device("mps", "Apple GPU · MPS"))
    devices.append(Device("cpu", f"CPU · {platform.machine()}"))
    return devices


@dataclass(frozen=True)
class Settings:
    confidence: float = 0.4
    opacity: float = 0.55
    color: str = "#20d7f5"
    output_mode: str = "overlay"
    tracking: bool = True
    device: str = "auto"
    image_size: int = 640


@dataclass(frozen=True)
class FrameResult:
    image: np.ndarray
    people: int
    track_ids: tuple[int, ...]
    inference_ms: float


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (245, 215, 32)
    try:
        return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))
    except ValueError:
        return (245, 215, 32)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return hex_to_bgr(TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)])


def compose_masks(frame, result, opacity, color, output_mode, track_ids=None):
    output = np.zeros_like(frame) if output_mode == "mask" else frame.copy()
    if result.masks is None:
        return output, 0
    fallback = np.array(hex_to_bgr(color), dtype=np.uint8)
    boxes = result.boxes.xyxy.int().cpu().numpy() if result.boxes is not None else []
    for index, mask in enumerate(result.masks.data.cpu().numpy()):
        resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
        track_id = track_ids[index] if track_ids and index < len(track_ids) else None
        mask_color = np.array(color_for_track(track_id), dtype=np.uint8) if track_id is not None else fallback
        if output_mode == "mask":
            output[resized] = mask_color if track_id is not None else (255, 255, 255)
        else:
            output[resized] = (output[resized] * (1 - opacity) + mask_color * opacity).astype(np.uint8)
        if track_id is not None and index < len(boxes):
            x1, y1, _, _ = boxes[index]
            label = f"ID {track_id}"
            size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            height = size[1] + baseline + 10
            left = max(0, min(int(x1), frame.shape[1] - size[0] - 14))
            top = max(0, int(y1) - height)
            bottom = min(frame.shape[0] - 1, top + height)
            cv2.rectangle(output, (left, top), (left + size[0] + 14, bottom), color_for_track(track_id), -1)
            cv2.putText(output, label, (left + 7, bottom - baseline - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 13, 15), 2, cv2.LINE_AA)
    return output, len(result.masks.data)


def encode_video(source: Path, destination: Path) -> None:
    executable = shutil.which("ffmpeg") or next((path for path in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if Path(path).is_file()), "ffmpeg")
    try:
        subprocess.run([executable, "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], check=True, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError("Install FFmpeg to export MP4 videos.") from error
    except subprocess.CalledProcessError as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Video encoding failed: {error.stderr.decode(errors='replace')}") from error


class SegmentationEngine:
    def __init__(self, device="auto", model_path=None):
        devices = available_devices()
        self.device = devices[0] if device == "auto" else next((item for item in devices if item.key == device), None)
        if self.device is None:
            raise RuntimeError(f"The selected inference device is unavailable: {device}")
        self.model = YOLO(str(model_path or ROOT / "yolo11n-seg.pt"))

    def warmup(self):
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), device=self.device.key, classes=[0], verbose=False)

    def reset(self):
        for tracker in getattr(getattr(self.model, "predictor", None), "trackers", []):
            tracker.reset()

    def process(self, frame: np.ndarray, settings: Settings, tracking=False) -> FrameResult:
        started = time.perf_counter()
        arguments = dict(classes=[0], conf=settings.confidence, device=self.device.key, imgsz=settings.image_size, verbose=False)
        if tracking:
            result = self.model.track(frame, persist=True, tracker=str(TRACKER_CONFIG), **arguments)[0]
            ids = () if result.boxes.id is None else tuple(result.boxes.id.int().cpu().tolist())
        else:
            result = self.model(frame, **arguments)[0]
            ids = ()
        output, people = compose_masks(frame, result, settings.opacity, settings.color, settings.output_mode, ids)
        return FrameResult(output, people, ids, (time.perf_counter() - started) * 1000)
