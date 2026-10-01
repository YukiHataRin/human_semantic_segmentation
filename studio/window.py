from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer, QSize, QRect, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from .engine import Settings, available_devices
from .workers import ProcessingWorker
from .cameras import CameraDiscovery, discover_cameras


class Preview(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.setMinimumSize(480, 320)
        self.setObjectName("preview")

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self.image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def clear(self):
        self.image = QImage()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#263943"))
        painter.setBrush(QColor("#101b23"))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 14, 14)
        if not self.image.isNull():
            size = self.image.size().scaled(self.size() - QSize(24, 24), Qt.AspectRatioMode.KeepAspectRatio)
            x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
            painter.drawImage(QRect(x, y, size.width(), size.height()), self.image)
        else:
            painter.setPen(QColor("#edf5f7"))
            painter.setFont(QFont(QApplication.font().family(), 23, QFont.Weight.DemiBold))
            painter.drawText(self.rect().adjusted(0, -34, 0, -34), Qt.AlignmentFlag.AlignCenter, "See every person.")
            painter.setPen(QColor("#96aab6"))
            painter.setFont(QFont(QApplication.font().family(), 12))
            painter.drawText(self.rect().adjusted(20, 40, -20, 40), Qt.AlignmentFlag.AlignCenter, "Select a camera and start live tracking.")


class MainWindow(QMainWindow):
    cameras_loaded = Signal()

    def __init__(self, worker_factory=ProcessingWorker, camera_discover=discover_cameras):
        super().__init__()
        self.worker_factory = worker_factory
        self.worker = None
        self.camera_discover = camera_discover
        self.camera_discovery = None
        self.last_frame = None
        self.source = None
        self.color = "#20d7f5"
        self.close_requested = False
        self.setWindowTitle("Human Mask Studio")
        self.resize(1240, 820)
        self.setMinimumSize(960, 640)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 18)
        layout.setSpacing(20)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        title_stack.setSpacing(4)
        title = QLabel("●  Human Mask Studio")
        title.setObjectName("brand")
        subtitle = QLabel("PERSON SEGMENTATION  /  LIVE ID TRACKING")
        subtitle.setObjectName("eyebrow")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch()
        layout.addLayout(header)

        workspace = QHBoxLayout()
        workspace.setSpacing(24)
        canvas = QVBoxLayout()
        canvas.setSpacing(12)
        self.preview = Preview()
        canvas.addWidget(self.preview, 1)
        captions = QHBoxLayout()
        self.filename = QLabel("Camera preview")
        self.filename.setObjectName("caption")
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        captions.addWidget(self.filename, 1)
        self.dimensions = QLabel("")
        self.dimensions.setObjectName("caption")
        captions.addWidget(self.dimensions)
        canvas.addLayout(captions)
        workspace.addLayout(canvas, 1)

        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setMinimumWidth(268)
        controls = QVBoxLayout(inspector)
        controls.setContentsMargins(20, 20, 20, 20)
        controls.setSpacing(9)
        self.camera_controls = QWidget()
        self.camera_controls.setObjectName("cameraControls")
        camera_layout = QVBoxLayout(self.camera_controls)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.addWidget(QLabel("Camera"))
        self.camera_devices = QComboBox()
        self.camera_devices.setAccessibleName("Camera device")
        self.camera_devices.setMinimumContentsLength(12)
        self.camera_devices.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        camera_layout.addWidget(self.camera_devices)
        self.camera_refresh = QPushButton("Refresh cameras")
        self.camera_refresh.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(self.camera_refresh)
        self.camera_status = QLabel("Finding cameras…")
        self.camera_status.setWordWrap(True)
        camera_layout.addWidget(self.camera_status)
        controls.addWidget(self.camera_controls)
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_camera)
        controls.addWidget(self.start_button)

        self.device = QComboBox()
        self.device.setAccessibleName("Inference device")
        devices = available_devices()
        self.device.addItem(f"Auto · {devices[0].label}", "auto")
        for device in devices:
            self.device.addItem(device.label, device.key)
        self.add_control(controls, "Inference device", self.device)
        self.device_label = QLabel(devices[0].label)
        self.device_label.setObjectName("deviceLabel")
        self.device_label.setWordWrap(True)
        controls.addWidget(self.device_label)
        model = QLabel("YOLO11n Segment")
        model.setObjectName("modelLabel")
        self.add_control(controls, "Model", model)

        self.tracking = QCheckBox("Track person IDs · BoT-SORT")
        self.tracking.setChecked(True)
        self.tracking.toggled.connect(self.settings_changed)
        controls.addWidget(self.tracking)
        self.confidence, self.confidence_value = self.add_slider(controls, "Confidence", 10, 90, 40)
        self.opacity, self.opacity_value = self.add_slider(controls, "Overlay opacity", 10, 90, 55)
        self.output_mode = QComboBox()
        self.output_mode.addItem("Mask overlay", "overlay")
        self.output_mode.addItem("Mask only", "mask")
        self.output_mode.currentIndexChanged.connect(self.settings_changed)
        self.add_control(controls, "Output", self.output_mode)
        self.color_button = QPushButton("Mask color · #20D7F5")
        self.color_button.clicked.connect(self.pick_color)
        controls.addWidget(self.color_button)
        controls.addStretch()

        self.metric_labels = {}
        for key, text in (("inference", "Inference"), ("people", "People"), ("tracks", "Track IDs"), ("fps", "Processing FPS")):
            row = QHBoxLayout()
            label = QLabel(text)
            label.setObjectName("metricName")
            value = QLabel("—")
            value.setObjectName("metricValue")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            controls.addLayout(row)
            self.metric_labels[key] = value
        inspector_scroll = QScrollArea()
        inspector_scroll.setObjectName("inspectorScroll")
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setFixedWidth(288)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        inspector_scroll.setWidget(inspector)
        workspace.addWidget(inspector_scroll)
        layout.addLayout(workspace, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("Ready for camera")
        self.status_label.setObjectName("status")
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        footer.addWidget(self.stop_button)
        self.save_button = QPushButton("Save snapshot…")
        self.save_button.clicked.connect(self.save_snapshot)
        self.save_button.setEnabled(False)
        footer.addWidget(self.save_button)
        layout.addLayout(footer)

        file_menu = self.menuBar().addMenu("File")
        self.save_action = QAction("Save snapshot…", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_snapshot)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        close_action = QAction("Close", self)
        close_action.setShortcut("Ctrl+Q")
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.refresh_preview)
        self.timer.start()
        QTimer.singleShot(0, self.refresh_cameras)

    def refresh_cameras(self):
        if self.camera_discovery or self.worker:
            return
        self.camera_status.setText("Finding cameras…")
        self.camera_discovery = CameraDiscovery(self, self.camera_discover)
        self.camera_discovery.ready.connect(self.on_cameras)
        self.camera_discovery.failed.connect(self.on_camera_discovery_error)
        self.camera_discovery.finished.connect(self.on_camera_discovery_finished)
        self.update_camera_controls()
        self.camera_discovery.start()

    def on_cameras(self, cameras):
        selected = self.camera_devices.currentData()
        self.camera_devices.clear()
        for camera in cameras:
            self.camera_devices.addItem(camera.name, camera)
            self.camera_devices.setItemData(self.camera_devices.count() - 1, camera.name, Qt.ItemDataRole.ToolTipRole)
        if selected:
            for index, camera in enumerate(cameras):
                if camera.uid == selected.uid:
                    self.camera_devices.setCurrentIndex(index)
                    break
        self.camera_status.setText(f"{len(cameras)} camera{'s' if len(cameras) != 1 else ''} detected" if cameras else "No cameras detected")
        self.cameras_loaded.emit()

    def on_camera_discovery_error(self, message):
        self.camera_devices.clear()
        self.camera_status.setText(message)

    def on_camera_discovery_finished(self):
        discovery, self.camera_discovery = self.camera_discovery, None
        discovery.deleteLater()
        self.update_camera_controls()
        if self.close_requested:
            self.close()

    def update_camera_controls(self):
        idle = self.worker is None
        ready = idle and self.camera_discovery is None
        self.camera_devices.setEnabled(ready and self.camera_devices.count() > 0)
        self.camera_refresh.setEnabled(ready)
        self.start_button.setEnabled(ready and self.camera_devices.count() > 0)

    def start_named_camera(self, name=""):
        for index in range(self.camera_devices.count()):
            camera = self.camera_devices.itemData(index)
            if not name or name in (camera.name, camera.uid):
                self.camera_devices.setCurrentIndex(index)
                self.launch(camera)
                return
        self.on_error(f"Camera not found: {name}" if name else "No cameras detected")

    def add_control(self, layout, text, widget):
        label = QLabel(text)
        label.setObjectName("controlLabel")
        layout.addWidget(label)
        layout.addWidget(widget)

    def add_slider(self, layout, text, low, high, value):
        row = QHBoxLayout()
        row.addWidget(QLabel(text))
        row.addStretch()
        output = QLabel(f"{value / 100:.2f}")
        output.setObjectName("sliderValue")
        row.addWidget(output)
        layout.addLayout(row)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setAccessibleName(text)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.setSingleStep(5)
        slider.valueChanged.connect(lambda number: output.setText(f"{number / 100:.2f}"))
        slider.valueChanged.connect(self.settings_changed)
        layout.addWidget(slider)
        return slider, output

    def settings(self):
        return Settings(confidence=self.confidence.value() / 100, opacity=self.opacity.value() / 100,
                        color=self.color, output_mode=self.output_mode.currentData(),
                        tracking=self.tracking.isChecked(), device=self.device.currentData())

    def settings_changed(self, *args):
        if self.worker:
            self.worker.update_settings(self.settings())

    def pick_color(self):
        color = QColorDialog.getColor(QColor(self.color), self, "Mask color")
        if color.isValid():
            self.color = color.name()
            self.color_button.setText(f"Mask color · {self.color.upper()}")
            self.settings_changed()

    def start_camera(self):
        camera = self.camera_devices.currentData()
        if camera is not None:
            self.launch(camera)

    def launch(self, source):
        if self.worker:
            return
        self.source = source
        self.last_frame = None
        self.save_button.setEnabled(False)
        self.save_action.setEnabled(False)
        self.preview.clear()
        for label in self.metric_labels.values():
            label.setText("—")
        self.filename.setText(source.name)
        self.filename.setToolTip(source.name)
        self.dimensions.clear()
        self.status_label.setText("Starting…")
        self.worker = self.worker_factory("camera", source, self.settings(), self)
        self.worker.status.connect(self.status_label.setText)
        self.worker.device_ready.connect(self.device_label.setText)
        self.worker.failed.connect(self.on_error)
        self.worker.completed.connect(self.on_complete)
        self.worker.finished.connect(self.on_finished)
        self.set_busy(True)
        self.worker.start()

    def set_busy(self, busy):
        self.device.setEnabled(not busy)
        self.update_camera_controls()
        self.stop_button.setEnabled(busy)

    def refresh_preview(self):
        if not self.worker:
            return
        preview = self.worker.take_preview()
        if preview is None:
            return
        result, metrics = preview
        self.last_frame = result.image
        self.preview.set_frame(result.image)
        height, width = result.image.shape[:2]
        self.dimensions.setText(f"{width} × {height}")
        self.metric_labels["inference"].setText(f"{result.inference_ms:.1f} ms")
        self.metric_labels["people"].setText(str(result.people))
        self.metric_labels["tracks"].setText(str(metrics["tracks"]))
        self.metric_labels["fps"].setText(f"{metrics['fps']:.1f}")
        self.save_button.setEnabled(True)
        self.save_action.setEnabled(True)

    def on_complete(self, payload):
        self.refresh_preview()
        self.status_label.setText("Stopped" if payload["stopped"] else "Camera finished")
        self.save_button.setEnabled(self.last_frame is not None)
        self.save_action.setEnabled(self.last_frame is not None)

    def on_error(self, message):
        self.status_label.setText(message)
        self.status_label.setToolTip(message)

    def on_finished(self):
        self.refresh_preview()
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.set_busy(False)
        if self.close_requested:
            self.close()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping…")

    def save_snapshot(self):
        if self.last_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save snapshot", "person-mask.png", "PNG image (*.png);;JPEG image (*.jpg)")
        if not path:
            return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(".png")
        try:
            if not cv2.imwrite(str(destination), self.last_frame):
                raise RuntimeError("Could not save this image.")
            self.status_label.setText(f"Saved · {destination.name}")
        except (OSError, RuntimeError, cv2.error) as error:
            self.on_error(str(error))

    def closeEvent(self, event):
        if self.worker or self.camera_discovery:
            self.close_requested = True
            self.stop()
            event.ignore()
        else:
            event.accept()
