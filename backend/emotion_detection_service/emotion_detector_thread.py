import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from skimage.feature import local_binary_pattern

import emotion_detection_service.globals as globals_module
from emotion_detection_service.exception import CustomException
from emotion_detection_service.logger import logging
from emotion_detection_service.predict import predict_emotion
from extensions import db
from models import DetectionLog, Camera, CameraStatus

FACE_DETECTOR_PROTO = "emotion_detection_service/deploy.prototxt"
FACE_DETECTOR_MODEL = "emotion_detection_service/res10_300x300_ssd_iter_140000.caffemodel"

IST = ZoneInfo("Asia/Kolkata")  # Indian Standard Time


class EmotionDetectorThread(threading.Thread):
    # At the beginning of the EmotionDetectorThread.__init__ method
    def __init__(self, cam_id, src, model_path, app):
        super().__init__()
        self.cam_id = cam_id
        self.src = src
        self.running = True
        self.last_negative_emotion = None
        self.app = app
        self.raw_frame = None
        self.processed_frame = None
        self.no_face_counter = 0
        self.no_face_threshold = 5  # You can tune this threshold
        self.emotion_buffer = deque(maxlen=10)  # last 10 frames
        self.negative_emotions = {'fear', 'anger', 'sadness', 'disgust'}
        self.sustain_threshold = 0.7  # 70% of last 10 frames

        # Check if CUDA is available for OpenCV
        self.has_cv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.has_cv_cuda:
            logging.info(f"OpenCV CUDA support enabled for camera {cam_id}")
            # Create CUDA streams for parallel processing
            self.cuda_stream = cv2.cuda.Stream()

        # Load face detector with GPU support if available
        self.face_net = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_PROTO, FACE_DETECTOR_MODEL)
        if self.has_cv_cuda:
            self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        logging.debug("Face detector loaded.")

        if self.src.startswith("rtsp://") and "rtsp_transport=tcp" not in self.src:
            self.src += "?rtsp_transport=tcp"

        self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Keep it minimal to reduce lag

        self.alerts_dir = os.path.abspath('static/alerts')
        os.makedirs(self.alerts_dir, exist_ok=True)

        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.model_path = model_path

        self.failure_count = 0
        self.max_failures = 20

        # Frame grabbing thread
        self.grab_running = True
        self.grab_thread = threading.Thread(target=self._frame_grabber)
        self.grab_thread.daemon = True
        self.grab_thread.start()

    def is_valid_face(self, face_img):
        """
        Validate if the detected region is actually a face using multiple checks
        """
        h, w = face_img.shape[:2]

        # Check 1: Minimum size filter
        if h < 40 or w < 40:
            logging.debug(f"Face too small: {w}x{h}")
            return False

        # Check 2: Aspect ratio check (faces are roughly square to slightly rectangular)
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            logging.debug(f"Invalid aspect ratio: {aspect_ratio}")
            return False

        # Check 3: Check for sufficient variation in pixel values
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

        # Calculate standard deviation - faces should have good contrast
        std_dev = np.std(gray)
        if std_dev < 10:  # Too uniform, likely not a face
            logging.debug(f"Low contrast region, std_dev: {std_dev}")
            return False

        # Check 4: Edge density check - faces have good edge content
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        if edge_density < 0.015:  # Too few edges
            logging.debug(f"Low edge density: {edge_density}")
            return False

        # Check 5: Brightness check - avoid very dark or very bright regions
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 220:
            logging.debug(f"Extreme brightness: {mean_brightness}")
            return False

        # New check 6: Circularity test (faces are rarely perfect circles)
        contours, _ = cv2.findContours(cv2.Canny(gray, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.8:  # Reject perfect circles
                    logging.debug(f"Circular object rejected: {circularity:.2f}")
                    return False

        # New check 7: Texture analysis using LBP
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp, bins=59, density=True)
        texture_score = np.std(hist)
        if texture_score < 0.15:  # Faces have complex textures
            logging.debug(f"Low texture variation: {texture_score:.2f}")
            return False

        return True

    def detect_faces(self, frame, conf_threshold=0.7):
        h, w = frame.shape[:2]
        logging.debug(f"Input frame dimensions: width={w}, height={h}")

        # Preprocess frame for better detection in low quality
        enhanced_frame = self.enhance_frame_quality(frame)

        # Optimize blob creation and inference
        blob = cv2.dnn.blobFromImage(cv2.resize(enhanced_frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)

        # Use async inference if CUDA is available
        if hasattr(self, 'has_cv_cuda') and self.has_cv_cuda:
            self.face_net.forward(outBlobNames=None, outputBlobs=None)
            detections = self.face_net.forward()
        else:
            detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            logging.debug(f"Detection {i}: confidence={confidence:.4f}")

            # Increase confidence threshold to reduce false positives
            if confidence > conf_threshold:  # Use higher threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Clip to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # Extract face region for validation
                face_region = frame[y1:y2, x1:x2]

                # Skip empty regions
                if face_region.size == 0:
                    continue

                # Validate if it's actually a face
                if self.is_valid_face(face_region):
                    faces.append((x1, y1, x2 - x1, y2 - y1))
                    logging.debug(f"Valid face detected with box coordinates: {(x1, y1, x2, y2)}")
                else:
                    logging.debug(f"Invalid face rejected at coordinates: {(x1, y1, x2, y2)}")

        logging.info(f"Total valid faces detected: {len(faces)}")
        return faces

    def enhance_frame_quality(self, frame):
        """
        Enhance frame quality for better face detection in low quality streams
        """
        # Convert to grayscale for processing
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # Apply slight Gaussian blur to reduce noise
        enhanced_gray = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)

        # Convert back to BGR if original was color
        if len(frame.shape) == 3:
            enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_frame = enhanced_gray

        return enhanced_frame

    def _frame_grabber(self):
        logging.info("Frame grabber thread started.")
        while self.grab_running:
            ret, frame = self.capture.read()
            if ret:
                with self.frame_lock:
                    self.raw_frame = frame.copy()
                self.failure_count = 0
                logging.debug("Frame grabbed successfully.")
            else:
                self.failure_count += 1
                logging.warning(f"Failed to grab frame. Failure count: {self.failure_count}")
                if self.failure_count >= self.max_failures:
                    logging.error("Max failure count reached, attempting to reconnect.")
                    self.reconnect()
                    self.failure_count = 0
            time.sleep(0.03)
        logging.info("Frame grabber thread stopped.")

    def get_latest_frame(self):
        with self.frame_lock:
            if self.processed_frame is not None:
                logging.debug("Returning a copy of the latest processed frame.")
                return self.processed_frame.copy()
            else:
                logging.debug("No processed frame available to return.")
                return None

    def save_alert(self, face_img, emotion, confidence):
        now_ist = datetime.now(IST)
        timestamp = now_ist.strftime("%Y%m%d_%H%M%S")
        filename = f"alert_{emotion}_{timestamp}.jpg"

        # Full path for saving image to disk
        filepath = os.path.join(self.alerts_dir, filename)
        cv2.imwrite(filepath, face_img)

        logging.info(f"Alert saved: {filepath}")
        logging.debug(f"IST time: {datetime.now(IST)}")

        # Store only relative path in DB
        relative_image_url = f"/static/alerts/{filename}"

        with self.app.app_context():
            try:
                camera = db.session.get(Camera, self.cam_id)
                camera_label = camera.label if camera else "Unknown"

                alert = DetectionLog(
                    camera_id=self.cam_id,
                    camera_label=camera_label,
                    timestamp=now_ist,
                    emotion=emotion,
                    confidence=confidence,
                    image_path=relative_image_url  # store relative URL
                )
                db.session.add(alert)
                db.session.commit()
                logging.info("Alert logged to DB.")
            except Exception as e:
                db.session.rollback()
                logging.error(f"Failed to log alert to DB: {e}")
                raise CustomException(e, sys)

    def reconnect(self):
        logging.warning(f"Reconnecting camera {self.cam_id}")
        self.capture.release()
        time.sleep(2)
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)

        # Check if reconnection worked
        if not self.capture.isOpened():
            logging.error(f"Failed to reconnect camera {self.cam_id}")
            self.failure_count = 0

            with self.app.app_context():
                try:
                    # Mark the camera as inactive in DB
                    camera = db.session.get(Camera, self.cam_id)
                    if camera:
                        camera.status = CameraStatus.Inactive
                        db.session.commit()
                        logging.info(f"Marked camera {self.cam_id} as Inactive due to repeated failures.")

                    # Stop and remove from manager
                    globals_module.manager.remove_camera(self.cam_id)

                except Exception as e:
                    db.session.rollback()
                    logging.error(f"Could not update camera status: {e}")
                    raise CustomException(e, sys)
        else:
            logging.info(f"Reconnected camera {self.cam_id}")
            self.failure_count = 0

    def run(self):

        logging.info(f"Starting emotion detection run loop for camera {self.cam_id}")

        while self.running:
            with self.frame_lock:
                frame = self.raw_frame.copy() if self.raw_frame is not None else None
            if frame is None:
                logging.debug("No frame available yet, sleeping briefly")
                time.sleep(0.01)
                continue

            faces = []
            if self.no_face_counter < self.no_face_threshold:
                faces = self.detect_faces(frame)
                logging.debug(f"Detected {len(faces)} faces")
                if not faces:
                    self.no_face_counter += 1
                    logging.debug(f"No faces detected, incrementing no_face_counter to {self.no_face_counter}")
                else:
                    self.no_face_counter = 0
            else:
                self.no_face_counter += 1
                logging.debug(f"Skipping face detection, no_face_counter={self.no_face_counter}")
                if self.no_face_counter > self.no_face_threshold + 3:
                    logging.debug("Resetting no_face_counter after skipping frames")
                    self.no_face_counter = 0

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]

                # Additional validation before emotion prediction
                if not self.is_valid_face(face_img):
                    logging.debug("Skipping emotion prediction for invalid face region")
                    continue

                emotion, confidence = predict_emotion(face_img, self.model_path)
                logging.debug(f"Predicted emotion: {emotion} with confidence {confidence:.2f}")

                # Increase confidence threshold for emotion prediction
                if confidence >= 0.65:  # Increased from 0.6 to 0.75
                    self.emotion_buffer.append(emotion)
                else:
                    self.emotion_buffer.append('neutral')  # treat low confidence as neutral

                # Increase sustain threshold to reduce false alarms
                negative_count = sum(1 for e in self.emotion_buffer if e in self.negative_emotions)
                ratio = negative_count / len(self.emotion_buffer)

                # Increased threshold from 0.7 to 0.8 (80% of frames must be negative)
                if ratio >= 0.7:
                    # Pick the most common negative emotion in buffer
                    from collections import Counter
                    counter = Counter(e for e in self.emotion_buffer if e in self.negative_emotions)

                    if counter:  # Make sure we have negative emotions
                        most_common_emotion, count = counter.most_common(1)[0]

                        # Additional check: require at least 6 out of 10 frames to be the same negative emotion
                        if count >= 5 and most_common_emotion != self.last_negative_emotion:
                            logging.info(
                                f"Sustained negative emotion detected: {most_common_emotion} with ratio {ratio:.2f}")
                            self.save_alert(face_img, most_common_emotion, confidence)
                            self.last_negative_emotion = most_common_emotion
                else:
                    self.last_negative_emotion = None

                # Draw rectangle and label as before, with current frame's emotion
                COLOR = (0, 255, 0)  # or use red for negative if you want to highlight
                THICKNESS = 2
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                FONT_SCALE = 0.8
                LINE_TYPE = cv2.LINE_AA

                cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR, THICKNESS, LINE_TYPE)
                label = f'{emotion}({confidence:.2f})'
                cv2.putText(frame, label, (x, y - 10), FONT, FONT_SCALE, COLOR, THICKNESS, LINE_TYPE)

            with self.frame_lock:
                self.processed_frame = frame

            time.sleep(0)

        logging.info(f"Stopping emotion detection run loop for camera {self.cam_id}")

        self.grab_running = False
        self.grab_thread.join()
        self.capture.release()

    @property
    def is_active(self):
        # Return True if the thread is running and the video capture is open
        return self.running and self.capture.isOpened()

    # Add this method to the EmotionDetectorThread class
    def cleanup_resources(self):
        # Release CUDA memory if using GPU
        if hasattr(self, 'has_cv_cuda') and self.has_cv_cuda:
            cv2.cuda.deviceReset()

        # Clear PyTorch cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Modify the stop method
    def stop(self):
        logging.info(f"Stopping camera processing for camera {self.cam_id}")
        self.running = False  # Stop the detection loop
        self.grab_running = False  # Stop the frame grabber loop
        if self.grab_thread.is_alive():
            self.grab_thread.join(timeout=5)
        if self.is_alive():  # wait for main thread if running
            self.join(timeout=5)
        self.capture.release()
        self.cleanup_resources()  # Add this line
        logging.info(f"Camera {self.cam_id} stopped and resources released")
