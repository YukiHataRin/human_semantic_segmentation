from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer, QSize, QRect, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
    QScrollArea, QSlider, QStyle, QTabBar, QVBoxLayout, QWidget,
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
            painter.drawText(self.rect().adjusted(20, 40, -20, 40), Qt.AlignmentFlag.AlignCenter, "Open an image or video, or start your camera.")


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
        self.result_path = None
        self.color = "#20d7f5"
        self.close_requested = False
        self.setWindowTitle("Human Mask Studio")
        self.resize(1240, 820)
        self.setMinimumSize(960, 640)
        self.setAcceptDrops(True)

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
        self.tabs = QTabBar()
        self.tabs.setObjectName("sourceTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        for label in ("Image", "Video", "Camera"):
            self.tabs.addTab(label)
        header.addWidget(self.tabs)
        layout.addLayout(header)

        workspace = QHBoxLayout()
        workspace.setSpacing(24)
        canvas = QVBoxLayout()
        canvas.setSpacing(12)
        self.preview = Preview()
        canvas.addWidget(self.preview, 1)
        captions = QHBoxLayout()
        self.filename = QLabel("No media selected")
        self.filename.setObjectName("caption")
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        captions.addWidget(self.filename, 1)
        self.dimensions = QLabel("")
        self.dimensions.setObjectName("caption")
        captions.addWidget(self.dimensions)
        canvas.addLayout(captions)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(5)
        self.progress.hide()
        canvas.addWidget(self.progress)
        workspace.addLayout(canvas, 1)

        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setMinimumWidth(268)
        controls = QVBoxLayout(inspector)
        controls.setContentsMargins(20, 20, 20, 20)
        controls.setSpacing(9)
        self.open_button = QPushButton("Open image…")
        self.open_button.setObjectName("primary")
        self.open_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_button.clicked.connect(self.choose_source)
        controls.addWidget(self.open_button)

        self.camera_controls = QWidget()
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
        self.camera_controls.hide()

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
        self.tracking.hide()
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
        self.apply_button = QPushButton("Apply to image")
        self.apply_button.clicked.connect(self.apply_image)
        self.apply_button.setEnabled(False)
        controls.addWidget(self.apply_button)
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
        self.status_label = QLabel("Ready for media")
        self.status_label.setObjectName("status")
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.hide()
        footer.addWidget(self.pause_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        footer.addWidget(self.stop_button)
        self.save_button = QPushButton("Save result…")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        footer.addWidget(self.save_button)
        layout.addLayout(footer)

        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open media…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_file)
        file_menu.addAction(open_action)
        save_action = QAction("Save result…", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        close_action = QAction("Close", self)
        close_action.setShortcut("Ctrl+Q")
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.refresh_preview)
        self.timer.start()
        self.tabs.currentChanged.connect(self.mode_changed)
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
        self.open_button.setEnabled(idle and (self.tabs.currentIndex() != 2 or (ready and self.camera_devices.count() > 0)))

    def start_named_camera(self, name=""):
        self.tabs.setCurrentIndex(2)
        for index in range(self.camera_devices.count()):
            camera = self.camera_devices.itemData(index)
            if not name or name in (camera.name, camera.uid):
                self.camera_devices.setCurrentIndex(index)
                self.launch("camera", camera)
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

    def mode_changed(self, index):
        self.open_button.setText(("Open image…", "Open video…", "Start camera")[index])
        self.camera_controls.setVisible(index == 2)
        self.tracking.setVisible(index != 0)
        self.apply_button.setVisible(index == 0)
        self.update_camera_controls()

    def pick_color(self):
        color = QColorDialog.getColor(QColor(self.color), self, "Mask color")
        if color.isValid():
            self.color = color.name()
            self.color_button.setText(f"Mask color · {self.color.upper()}")
            self.settings_changed()

    def choose_source(self):
        if self.tabs.currentIndex() == 2:
            camera = self.camera_devices.currentData()
            if camera is not None:
                self.launch("camera", camera)
        else:
            self.choose_file()

    def choose_file(self):
        if self.worker:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open image or video", "", "Media (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.mp4 *.mov *.avi *.mkv *.webm *.m4v);;All files (*)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        mode = "image" if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"} else "video"
        self.tabs.setCurrentIndex(0 if mode == "image" else 1)
        self.launch(mode, path)

    def launch(self, mode, source):
        if self.worker:
            return
        self.source = (mode, source)
        self.result_path = None
        self.last_frame = None
        self.preview.clear()
        for label in self.metric_labels.values():
            label.setText("—")
        self.filename.setText(source.name if mode == "camera" else Path(source).name)
        self.filename.setToolTip(source.name if mode == "camera" else str(source))
        self.dimensions.clear()
        self.status_label.setText("Starting…")
        self.worker = self.worker_factory(mode, source, self.settings(), self)
        self.worker.status.connect(self.status_label.setText)
        self.worker.device_ready.connect(self.device_label.setText)
        self.worker.failed.connect(self.on_error)
        self.worker.completed.connect(self.on_complete)
        self.worker.finished.connect(self.on_finished)
        self.set_busy(True)
        self.progress.setRange(0, 0)
        self.progress.setVisible(mode != "camera")
        self.pause_button.setVisible(mode == "video")
        self.worker.start()

    def set_busy(self, busy):
        for control in (self.tabs, self.device):
            control.setEnabled(not busy)
        self.update_camera_controls()
        self.stop_button.setEnabled(busy)
        self.pause_button.setEnabled(busy)
        self.apply_button.setEnabled(not busy and self.source is not None and self.source[0] == "image")

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
        if metrics["total"]:
            self.progress.setRange(0, metrics["total"])
            self.progress.setValue(metrics["frames"])
        self.save_button.setEnabled(self.source[0] != "video")

    def on_complete(self, payload):
        self.refresh_preview()
        self.result_path = payload["path"]
        self.status_label.setText("Stopped" if payload["stopped"] else "Image ready" if self.source[0] == "image" else "Video ready · saved to outputs")
        self.save_button.setEnabled(self.last_frame is not None and (self.source[0] != "video" or self.result_path is not None))

    def on_error(self, message):
        self.status_label.setText(message)
        self.status_label.setToolTip(message)

    def on_finished(self):
        self.refresh_preview()
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.set_busy(False)
        self.progress.hide()
        self.pause_button.hide()
        self.pause_button.setText("Pause")
        if self.close_requested:
            self.close()

    def apply_image(self):
        if self.source and self.source[0] == "image":
            self.launch(*self.source)

    def toggle_pause(self):
        if self.worker:
            if self.worker.paused.is_set():
                self.worker.paused.clear()
                self.pause_button.setText("Pause")
                self.status_label.setText("Processing video")
            else:
                self.worker.paused.set()
                self.pause_button.setText("Resume")
                self.status_label.setText("Paused")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.status_label.setText("Stopping…")

    def save_result(self):
        if self.last_frame is None or (self.source[0] == "video" and not self.result_path):
            return
        video = self.result_path is not None
        path, _ = QFileDialog.getSaveFileName(self, "Save result", self.result_path.name if video else "person-mask.png", "MP4 video (*.mp4)" if video else "PNG image (*.png);;JPEG image (*.jpg)")
        if not path:
            return
        destination = Path(path)
        if not destination.suffix:
            destination = destination.with_suffix(".mp4" if video else ".png")
        try:
            if video:
                if self.result_path.resolve() != destination.resolve():
                    shutil.copyfile(self.result_path, destination)
            elif not cv2.imwrite(str(destination), self.last_frame):
                raise RuntimeError("Could not save this image.")
            self.status_label.setText(f"Saved · {destination.name}")
        except (OSError, RuntimeError, cv2.error) as error:
            self.on_error(str(error))

    def dragEnterEvent(self, event):
        if not self.worker and event.mimeData().hasUrls() and event.mimeData().urls()[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not self.worker:
            self.open_path(event.mimeData().urls()[0].toLocalFile())

    def closeEvent(self, event):
        if self.worker or self.camera_discovery:
            self.close_requested = True
            self.stop()
            event.ignore()
        else:
            event.accept()
