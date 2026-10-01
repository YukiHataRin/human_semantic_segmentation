import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from studio.engine import ROOT, Device, FrameResult, Settings, SegmentationEngine, compose_masks
from studio.window import MainWindow
from studio.workers import CameraReader, LatestFrame, ProcessingWorker
from studio.cameras import Camera


class FakeCapture:
    def __init__(self, opened=True, disconnect=False):
        self.opened = opened
        self.disconnect = disconnect
        self.released = threading.Event()

    def isOpened(self):
        return self.opened

    def get(self, property):
        return 30

    def read(self):
        time.sleep(0.005)
        return (False, None) if self.disconnect else (True, np.zeros((48, 64, 3), dtype=np.uint8))

    def release(self):
        self.released.set()


class FakeEngine:
    def __init__(self, device):
        self.device = Device("cpu", "CPU · test")

    def warmup(self):
        pass

    def reset(self):
        pass

    def process(self, frame, settings, tracking=False):
        time.sleep(0.01)
        return FrameResult(frame.copy(), 1, (1,) if tracking else (), 10)


class EngineTests(unittest.TestCase):
    def test_auto_device_prefers_available_accelerator(self):
        with patch("studio.engine.available_devices", return_value=[Device("mps", "Apple GPU · MPS"), Device("cpu", "CPU")]), patch("studio.engine.YOLO"):
            self.assertEqual(SegmentationEngine().device.key, "mps")

    def test_unavailable_device_fails_without_loading_model(self):
        with patch("studio.engine.available_devices", return_value=[Device("cpu", "CPU")]), patch("studio.engine.YOLO") as model:
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                SegmentationEngine("cuda:0")
            model.assert_not_called()

    def test_no_detections_preserve_overlay_and_clear_mask(self):
        frame = np.full((32, 32, 3), 100, np.uint8)
        result = SimpleNamespace(masks=None)
        overlay, count = compose_masks(frame, result, 0.5, "#20d7f5", "overlay")
        np.testing.assert_array_equal(overlay, frame)
        mask, count = compose_masks(frame, result, 0.5, "#20d7f5", "mask")
        self.assertFalse(mask.any())
        self.assertEqual(count, 0)

    def test_mask_only_without_ids_is_white_inside_mask(self):
        mask = torch.zeros((1, 32, 32))
        mask[:, 10:20, 10:20] = 1
        result = SimpleNamespace(masks=SimpleNamespace(data=mask), boxes=None)
        output, count = compose_masks(np.zeros((32, 32, 3), np.uint8), result, 0.5, "#20d7f5", "mask")
        np.testing.assert_array_equal(output[15, 15], (255, 255, 255))
        self.assertFalse(output[0, 0].any())
        self.assertEqual(count, 1)


class CaptureTests(unittest.TestCase):
    def test_camera_waits_for_first_frame_during_startup(self):
        capture = FakeCapture()
        reads = 0
        original_read = capture.read
        def delayed_read():
            nonlocal reads
            reads += 1
            return (False, None) if reads < 3 else original_read()
        capture.read = delayed_read
        reader = CameraReader(capture, startup_timeout=1)
        reader.start()
        self.assertIsNotNone(reader.frames.take(timeout=1))
        reader.stop()
        reader.join(1)
        self.assertIsNone(reader.error)
        self.assertTrue(capture.released.is_set())

    def test_mailbox_drops_stale_frames(self):
        mailbox = LatestFrame()
        mailbox.put("old")
        mailbox.put("latest")
        self.assertEqual(mailbox.take(), "latest")
        self.assertIsNone(mailbox.take(0))
        mailbox.close()
        self.assertIsNone(mailbox.take(0))

    def test_disconnect_releases_camera(self):
        capture = FakeCapture(disconnect=True)
        reader = CameraReader(capture, startup_timeout=0.02)
        reader.start()
        reader.join(1)
        self.assertFalse(reader.is_alive())
        self.assertTrue(capture.released.is_set())
        self.assertTrue(reader.frames.closed)
        self.assertIn("camera", reader.error.lower())


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.capture = FakeCapture()
        self.cameras = [Camera("Built-in camera", 0, 1200, "built-in"), Camera("USB camera", 1, 1200, "usb")]
        self.window = MainWindow(worker_factory=lambda mode, source, settings, parent: ProcessingWorker(mode, source, settings, parent, FakeEngine, lambda *args, **kwargs: self.capture), camera_discover=lambda: self.cameras)
        self.window.show()
        self.wait_until(lambda: self.window.camera_devices.count() == 2 and self.window.camera_discovery is None)

    def wait_until(self, predicate, timeout=3000):
        deadline = time.monotonic() + timeout / 1000
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate(), "Timed out waiting for the native UI state")

    def tearDown(self):
        if self.window.worker:
            self.window.stop()
            self.wait_until(lambda: self.window.worker is None)
        self.window.close()
        self.app.processEvents()

    def start_camera(self):
        self.window.start_button.click()
        self.wait_until(lambda: self.window.last_frame is not None)

    def test_camera_stop_releases_device_and_allows_restart(self):
        self.start_camera()
        self.assertFalse(self.window.device.isEnabled())
        self.assertEqual(self.window.metric_labels["people"].text(), "1")
        self.window.stop_button.click()
        self.wait_until(lambda: self.window.worker is None)
        self.assertTrue(self.capture.released.is_set())
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertEqual(self.window.status_label.text(), "Stopped")
        self.capture = FakeCapture()
        self.start_camera()

    def test_camera_error_restores_controls(self):
        self.capture = FakeCapture(opened=False)
        self.window.launch(self.cameras[0])
        self.wait_until(lambda: self.window.worker is None)
        self.assertTrue(self.capture.released.is_set())
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertIn("Could not open the camera", self.window.status_label.text())

    def test_named_camera_selection_reaches_worker(self):
        self.window.start_named_camera("USB camera")
        self.wait_until(lambda: self.window.last_frame is not None)
        self.assertEqual(self.window.worker.source, self.cameras[1])
        self.assertEqual(self.window.filename.text(), "USB camera")

    def test_refresh_preserves_device_identity_when_index_changes(self):
        self.window.camera_devices.setCurrentIndex(1)
        self.cameras = [Camera("USB camera", 0, 1200, "usb")]
        self.window.refresh_cameras()
        self.wait_until(lambda: self.window.camera_discovery is None)
        self.assertEqual(self.window.camera_devices.currentData().uid, "usb")
        self.assertEqual(self.window.camera_devices.currentData().index, 0)

    def test_no_cameras_disables_start_and_allows_refresh(self):
        self.cameras = []
        self.window.refresh_cameras()
        self.wait_until(lambda: self.window.camera_discovery is None)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(self.window.camera_refresh.isEnabled())
        self.assertEqual(self.window.camera_status.text(), "No cameras detected")

    def test_discovery_failure_clears_stale_devices(self):
        def fail():
            raise RuntimeError("Discovery failed")
        self.window.camera_discover = fail
        self.window.refresh_cameras()
        self.wait_until(lambda: self.window.camera_discovery is None)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertEqual(self.window.camera_devices.count(), 0)
        self.assertIn("Discovery failed", self.window.camera_status.text())

    def test_live_settings_are_sent_to_worker(self):
        self.start_camera()
        self.window.confidence.setValue(65)
        self.window.tracking.setChecked(False)
        self.assertEqual(self.window.worker.settings().confidence, 0.65)
        self.assertFalse(self.window.worker.settings().tracking)

    def test_snapshot_saves_current_camera_frame_as_png(self):
        self.assertFalse(self.window.save_button.isEnabled())
        self.start_camera()
        self.window.stop_button.click()
        self.wait_until(lambda: self.window.worker is None)
        with TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            destination = Path(directory) / "camera snapshot"
            with patch("studio.window.QFileDialog.getSaveFileName", return_value=(str(destination), "PNG image (*.png)")):
                self.window.save_snapshot()
            import cv2
            saved = cv2.imread(str(destination.with_suffix(".png")))
            np.testing.assert_array_equal(saved, self.window.last_frame)
            self.assertEqual(self.window.status_label.text(), "Saved · camera snapshot.png")

    def test_closing_window_waits_for_camera_release(self):
        self.start_camera()
        self.window.close()
        self.wait_until(lambda: self.window.worker is None)
        self.assertTrue(self.capture.released.is_set())
        self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()
