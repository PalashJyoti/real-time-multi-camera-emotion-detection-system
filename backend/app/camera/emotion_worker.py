import datetime

from sqlalchemy.orm import scoped_session, sessionmaker

from extensions import db

Session = scoped_session(sessionmaker(bind=db.engine))

import threading
import time
import cv2
import os
from datetime import datetime
from app.camera.model import predict_emotion
from models import DetectionLog
from app import db
from logger import logging
from exception import CustomException
import sys

NEGATIVE_EMOTIONS = {'fear', 'anger', 'sadness', 'disgust'}
EMOTION_THRESHOLD = 0.5


class EmotionDetectorThread(threading.Thread):
    def __init__(self, cam_id, model_path, app):
        super().__init__()
        self.cam_id = cam_id
        self.model_path = model_path
        self.app = app
        self.running = True
        self.last_negative_emotion = None

        from app.camera.camera_manager import get_camera_manager
        self.camera_manager = get_camera_manager()

        self.alerts_dir = os.path.abspath('static/alerts')
        os.makedirs(self.alerts_dir, exist_ok=True)
        logging.debug(f"[Camera {self.cam_id}] Alerts directory ensured at {self.alerts_dir}")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.latest_emotion = None
        self.latest_confidence = None
        self.frame_lock = threading.Lock()
        self.frame = None

        logging.info(f"[Camera {self.cam_id}] EmotionDetectorThread initialized")

    def get_latest_frame(self):
        with self.frame_lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
            if self.latest_emotion and self.latest_confidence is not None:
                text = f"{self.latest_emotion} ({self.latest_confidence:.2f})"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame

    def run(self):
        logging.info(f"[Camera {self.cam_id}] Starting EmotionDetectorThread")

        detection_interval = 0.5
        last_time = time.monotonic()

        while self.running:
            now = time.monotonic()
            elapsed = now - last_time
            if elapsed < detection_interval:
                time.sleep(detection_interval - elapsed)
            last_time = time.monotonic()

            frame = self._get_frame()
            if frame is None:
                logging.warning(f"[Camera {self.cam_id}] No frame received")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                with self.frame_lock:
                    self.latest_emotion = None
                    self.latest_confidence = None
                    self.frame = frame
                logging.debug(f"[Camera {self.cam_id}] No face detected")
                continue

            (x, y, w, h) = faces[0]
            face_img = frame[y:y + h, x:x + w]

            emotion, confidence = predict_emotion(face_img, self.model_path)
            logging.debug(f"[Camera {self.cam_id}] Detected emotion: {emotion} with confidence {confidence:.2f}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 15)
            label = f"{emotion} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.0
            thickness = 15
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x + 5
            text_y = y + text_size[1] + 5
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

            with self.frame_lock:
                self.latest_emotion = emotion
                self.latest_confidence = confidence
                self.frame = frame

            if emotion in NEGATIVE_EMOTIONS and confidence >= EMOTION_THRESHOLD:
                if emotion != self.last_negative_emotion:
                    self._save_snapshot_and_log(frame, emotion, confidence)
                    self.last_negative_emotion = emotion
            else:
                self.last_negative_emotion = None

        logging.info(f"[Camera {self.cam_id}] EmotionDetectorThread stopped")

    def _get_frame(self):
        cam = self.camera_manager.cameras.get(self.cam_id)
        if cam is None:
            logging.warning(f"[Camera {self.cam_id}] Camera stream not found")
            return None
        frame = cam.get_frame()
        if frame is None:
            logging.warning(f"[Camera {self.cam_id}] Camera stream returned no frame")
        return frame

    def _save_snapshot_and_log(self, frame, emotion, confidence):
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"alert_{emotion}_{timestamp_str}.jpg"
        full_path = os.path.join(self.alerts_dir, filename)
        relative_path = f"static/alerts/{filename}"

        cv2.imwrite(full_path, frame)
        logging.info(f"[Camera {self.cam_id}] Snapshot saved at {full_path}")

        with self.app.app_context():
            try:
                log = DetectionLog(
                    camera_id=self.cam_id,
                    emotion=emotion,
                    confidence=confidence,
                    image_path=relative_path,
                    timestamp=timestamp
                )
                db.session.add(log)
                db.session.commit()
                logging.info(f"[Camera {self.cam_id}] Detection logged in database: {emotion} ({confidence:.2f})")
            except Exception as e:
                logging.exception(f"[Camera {self.cam_id}] Failed to log detection: {e}")
                raise CustomException(e, sys)

    def stop(self):
        logging.info(f"[Camera {self.cam_id}] Stopping EmotionDetectorThread")
        self.running = False
