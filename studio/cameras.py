from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True)
class Camera:
    name: str
    index: int
    backend: int
    uid: str


def enumerate_devices():
    from cv2_enumerate_cameras import enumerate_cameras

    backend = {"darwin": cv2.CAP_AVFOUNDATION, "win32": cv2.CAP_MSMF}.get(sys.platform, cv2.CAP_V4L2)
    return [Camera(item.name or "Camera", item.index, item.backend,
                   item.path or f"{item.backend}:{item.index}")
            for item in enumerate_cameras(backend)]


def discover_cameras():
    # A fresh process also refreshes AVFoundation's device inventory after hotplug.
    result = subprocess.run(
        [sys.executable, "-m", "studio.cameras"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True, timeout=10, check=True,
    )
    return [Camera(**item) for item in json.loads(result.stdout)]


class CameraDiscovery(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None, discover=discover_cameras):
        super().__init__(parent)
        self.discover = discover

    def run(self):
        try:
            self.ready.emit(self.discover())
        except Exception as error:
            self.failed.emit(f"Could not list cameras: {error}")


if __name__ == "__main__":
    print(json.dumps([asdict(camera) for camera in enumerate_devices()]))
