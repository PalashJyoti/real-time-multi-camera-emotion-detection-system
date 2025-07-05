import threading
import time

import cv2

from logger import logging
from models import Camera, CameraStatus


def get_active_camera_sources():
    logging.info("Fetching active camera sources from database")

    cameras = Camera.query.filter_by(status=CameraStatus.Active).all()
    sources = [(cam.id, cam.src) for cam in cameras]

    logging.debug(f"Active cameras found: {len(sources)}")
    return sources


class CameraStream:
    def __init__(self, src, camera_id):
        self.src = src
        self.camera_id = camera_id
        self.capture = cv2.VideoCapture(src)
        self.frame = None
        self.running = False
        self.valid = self.capture.isOpened()

        if self.valid:
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            logging.info(f"✅ Camera {camera_id} started with src: {src}")
        else:
            logging.error(f"❌ Failed to open camera {camera_id} with src: {src}")

    def update(self):
        failure_count = 0
        max_failures = 45  # 3 seconds of continuous failure at 15 FPS

        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame = frame
                failure_count = 0
            else:
                failure_count += 1
                if self.capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                elif failure_count >= max_failures:
                    logging.warning(
                        f"⚠️ Camera {self.camera_id} read failed for 3 seconds straight. Stream might be dead."
                    )
            time.sleep(1 / 15)

    def get_frame(self):
        return self.frame if self.valid else None

    def stop(self):
        self.running = False
        if self.capture.isOpened():
            self.capture.release()
        logging.info(f"Camera {self.camera_id} stopped and released")


class MultiCameraManager:
    def __init__(self, camera_sources):
        self.cameras = {}

        for cam_id, src in camera_sources:
            cam_stream = CameraStream(src, cam_id)
            if cam_stream.valid:
                self.cameras[cam_id] = cam_stream
                logging.info(f"✅ Camera {cam_id} initialized and running.")
            else:
                logging.error(f"❌ Skipping invalid camera source: {src} (ID: {cam_id})")

    def get_frame(self, camera_id):
        cam = self.cameras.get(camera_id)
        return cam.get_frame() if cam else None

    def stop_camera(self, camera_id):
        cam = self.cameras.get(camera_id)
        if cam:
            cam.stop()
            if hasattr(cam, 'join'):
                cam.join(timeout=2)
            del self.cameras[camera_id]
            logging.info(f"🛑 Camera {camera_id} stopped and removed.")

    def stop_all(self):
        logging.info("Stopping all camera streams")
        for cam_id in list(self.cameras.keys()):
            self.stop_camera(cam_id)

    def restart_camera(self, camera_id, src):
        if camera_id in self.cameras:
            logging.info(f"🔄 Restarting camera {camera_id}")
            self.stop_camera(camera_id)

        new_cam_stream = CameraStream(src, camera_id)
        if new_cam_stream.valid:
            self.cameras[camera_id] = new_cam_stream
            logging.info(f"✅ Camera {camera_id} restarted successfully.")
        else:
            logging.error(f"❌ Failed to restart camera {camera_id} with source: {src}")


# Initialize
_camera_manager = None  # global singleton variable


def get_camera_manager():
    global _camera_manager
    if _camera_manager is None:
        logging.info("Initializing MultiCameraManager with empty camera source list")
        _camera_manager = MultiCameraManager(camera_sources=[])
        _camera_manager.emotion_detectors = {}
        logging.info("MultiCameraManager initialized successfully")
    return _camera_manager


def set_camera_manager(manager):
    global _camera_manager
    _camera_manager = manager
    logging.info("Camera manager has been set via set_camera_manager")


def init_camera_manager():
    sources = get_active_camera_sources()
    logging.info(f"Initializing camera manager with sources: {sources}")
    set_camera_manager(MultiCameraManager(sources))
