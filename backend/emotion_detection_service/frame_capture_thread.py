import sys
import threading
import time

import cv2

from emotion_detection_service.exception import CustomException
from emotion_detection_service.logger import logging


class FrameCaptureThread(threading.Thread):
    def __init__(self, src, cam_id):
        super().__init__()
        self.src = src
        self.cam_id = cam_id

        # Set buffer size to 1 to always get the latest frame
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to set optimal resolution and FPS
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for faster processing
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for RTSP streams to reduce load

        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.failure_count = 0
        self.max_failures = 20

        # Add frame skipping for performance
        self.frame_count = 0
        self.process_every_n_frames = 2  # Process every 2nd frame

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frame_count += 1
                # Only process every nth frame
                if self.frame_count % self.process_every_n_frames == 0:
                    with self.frame_lock:
                        self.latest_frame = frame
                self.failure_count = 0
            else:
                self.failure_count += 1
                logging.warning(f"[Camera {self.cam_id}] Frame grab failed ({self.failure_count})")
                if self.failure_count >= self.max_failures:
                    self.reconnect()
            # Reduce CPU usage with a small sleep
            time.sleep(0.01)

        self.capture.release()

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def reconnect(self):
        logging.info(f"[Camera {self.cam_id}] Reconnecting to stream...")
        try:
            self.capture.release()
            time.sleep(1)
            if self.src.startswith("rtsp://") and "rtsp_transport=tcp" not in self.src:
                self.src += "?rtsp_transport=tcp"
            for attempt in range(3):
                self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                if self.capture.isOpened():
                    logging.info(f"[Camera {self.cam_id}] Reconnected on attempt {attempt + 1}")
                    break
                time.sleep(2)
            else:
                logging.error(f"[Camera {self.cam_id}] Failed to reconnect after 3 attempts.")
        except Exception as e:
            logging.exception(f"[Camera {self.cam_id}] Reconnect error: {e}")
            raise CustomException(e, sys)
        self.failure_count = 0

    def stop(self):
        self.running = False
