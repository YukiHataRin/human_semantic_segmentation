from __future__ import annotations

import argparse
import os
import sys

from .engine import ROOT

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from .window import MainWindow
from .cameras import discover_cameras


def create_application():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Human Mask Studio")
    app.setOrganizationName("Human Mask Studio")
    if sys.platform == "darwin":
        app.setFont(QFont("Helvetica Neue", 12))
    apply_stylesheet(app, theme="dark_teal.xml", extra={"font_family": app.font().family(), "density_scale": "-1"})
    app.setStyleSheet(app.styleSheet() + """
        QWidget#root { background: #0c151d; }
        QLabel { background: transparent; margin: 0; padding: 0; }
        QScrollArea#inspectorScroll { background: transparent; }
        QLabel#brand { font-size: 25px; font-weight: 650; color: #edf5f7; }
        QLabel#eyebrow { font-size: 10px; color: #78939f; letter-spacing: 2px; }
        QLabel#caption, QLabel#status { color: #96aab6; font-size: 12px; }
        QLabel#deviceLabel { color: #41dfc8; font-size: 12px; }
        QLabel#controlLabel, QLabel#metricName { color: #96aab6; font-size: 12px; }
        QLabel#metricValue { color: #edf5f7; font-size: 16px; font-weight: 600; }
        QLabel#modelLabel { color: #edf5f7; font-size: 14px; padding: 4px 0; }
        QLabel#sliderValue { color: #41dfc8; }
        QFrame#inspector { background: #14212b; border: 1px solid #263943; border-radius: 14px; }
        QPushButton { text-transform: none; min-height: 24px; border-radius: 7px; }
        QPushButton#primary { background: #41dfc8; color: #0c151d; font-weight: 600; }
        QPushButton#primary:disabled { background: #263943; color: #78939f; }
        QTabBar#sourceTabs::tab { padding: 10px 24px; margin: 0; background: #14212b; border: 0; border-radius: 7px; color: #96aab6; text-transform: none; }
        QTabBar#sourceTabs::tab:selected { background: #204d50; color: #41dfc8; }
        QProgressBar { border: 0; background: #14212b; }
        QProgressBar::chunk { background: #41dfc8; }
    """)
    return app


def main():
    parser = argparse.ArgumentParser(description="Human Mask Studio native desktop application")
    parser.add_argument("media", nargs="?", help="Image or video to open")
    parser.add_argument("--camera", nargs="?", const="", help="Start a camera by its name or device ID; omit the name for the first camera")
    parser.add_argument("--list-cameras", action="store_true", help="List connected camera names and device IDs")
    args = parser.parse_args()
    if args.list_cameras:
        for camera in discover_cameras():
            print(f"{camera.name}\t{camera.uid}")
        return
    app = create_application()
    window = MainWindow()
    window.show()
    if args.camera is not None:
        window.tabs.setCurrentIndex(2)
        def start_requested_camera():
            window.cameras_loaded.disconnect(start_requested_camera)
            window.start_named_camera(args.camera)
        window.cameras_loaded.connect(start_requested_camera)
    elif args.media:
        QTimer.singleShot(0, lambda: window.open_path(args.media))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
