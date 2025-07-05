import base64
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image
from flask import Blueprint, send_from_directory, Response, stream_with_context, jsonify, request
from pytz import timezone as pytz_timezone, utc
from sqlalchemy.exc import IntegrityError

from app.camera.model import predict_emotion, ResEmoteNet
from exception import CustomException
from extensions import db
from ip import ipaddress
from logger import logging
from models import DetectionLog, Camera, CameraStatus

camera_bp = Blueprint('camera_feed', __name__)

IST = ZoneInfo("Asia/Kolkata")


# Helper function for resizing with padding (16:9)
def resize_and_pad(image, target_size):
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)

    logging.debug(
        f"[resize_and_pad] Original size: ({w}, {h}), Target size: ({target_w}, {target_h}), Scale: {scale:.4f}")

    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    logging.debug(f"[resize_and_pad] Resized size: {resized.shape[1]}x{resized.shape[0]}")

    top_pad = (target_h - resized.shape[0]) // 2
    bottom_pad = target_h - resized.shape[0] - top_pad
    left_pad = (target_w - resized.shape[1]) // 2
    right_pad = target_w - resized.shape[1] - left_pad

    logging.debug(
        f"[resize_and_pad] Padding (top: {top_pad}, bottom: {bottom_pad}, left: {left_pad}, right: {right_pad})")

    padded = cv2.copyMakeBorder(resized, top_pad, bottom_pad, left_pad, right_pad,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded


# Emotion-detected feed from background thread
@camera_bp.route('/api/camera_feed/<int:camera_id>')
def camera_feed(camera_id):
    emotion_service_url = f"http://{ipaddress}:5001/stream/{camera_id}"
    logging.debug(f"[camera_feed] Starting stream for camera_id={camera_id} from {emotion_service_url}")

    def external_stream():
        try:
            with requests.get(emotion_service_url, stream=True, timeout=5) as r:
                r.raise_for_status()
                logging.debug(f"[camera_feed] Connection successful to emotion service for camera_id={camera_id}")
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
        except requests.RequestException as e:
            logging.warning(f"[camera_feed] External stream failed for camera_id={camera_id}: {e}")
            # Raise to propagate the error and return HTTP 500
            raise CustomException(e, sys)

    return Response(stream_with_context(external_stream()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@camera_bp.route('/api/detection-logs', methods=['GET'])
def get_detection_logs():
    range_filter = request.args.get('range', 'overall').lower()
    now_ist = datetime.now(IST)  # Use IST directly

    logging.debug(f"[get_detection_logs] Range filter: {range_filter}")
    logging.debug(f"[get_detection_logs] Current IST time: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine time range filter
    if range_filter == 'weekly':
        start_time = now_ist - timedelta(days=7)
    elif range_filter == 'monthly':
        start_time = now_ist - timedelta(days=30)
    elif range_filter == 'yearly':
        start_time = now_ist - timedelta(days=365)
    elif range_filter == 'overall':
        start_time = None
    else:
        logging.warning(f"[get_detection_logs] Invalid range parameter: {range_filter}")
        return jsonify({"error": "Invalid range parameter"}), 400

    # Fetch logs from database
    if start_time:
        logs = DetectionLog.query.filter(
            DetectionLog.timestamp >= start_time
        ).order_by(DetectionLog.timestamp.desc()).all()
        logging.debug(f"[get_detection_logs] Logs fetched from {start_time.strftime('%Y-%m-%d %H:%M:%S')} to now.")
    else:
        logs = DetectionLog.query.order_by(
            DetectionLog.timestamp.desc()
        ).all()
        logging.debug(f"[get_detection_logs] Fetched all logs (no time filter).")

    logging.info(f"[get_detection_logs] Total logs returned: {len(logs)}")

    # Format logs
    result = []
    for log in logs:
        result.append({
            'id': log.id,
            'camera_label': getattr(log, 'camera_label', 'Unknown'),
            'emotion': log.emotion,
            'timestamp': log.timestamp.strftime('%b %d, %Y, %I:%M %p') if log.timestamp else None,
            'image_url': f"/{log.image_path}" if log.image_path else None
        })

    return jsonify(result)


# Serve alert screenshots
@camera_bp.route('/alerts/<path:filename>')
def serve_alert_image(filename):
    alerts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'alerts')
    full_path = os.path.join(alerts_dir, filename)

    logging.debug(f"[serve_alert_image] Request for: {filename}")
    logging.debug(f"[serve_alert_image] Full path: {full_path}")

    if not os.path.isfile(full_path):
        logging.warning(f"[serve_alert_image] File not found: {full_path}")

    return send_from_directory(alerts_dir, filename)


@camera_bp.route('/api/cameras', methods=['GET'])
def get_cameras():
    logging.debug("[get_cameras] Fetching all cameras from database.")

    cameras = Camera.query.all()
    logging.info(f"[get_cameras] Total cameras fetched: {len(cameras)}")

    return jsonify([camera.to_dict() for camera in cameras])


def get_detection_analytics():
    """
    Returns detection analytics (pie chart, timeline, trends, etc.)
    Filtered by time range: '5m', '30m', '1h', 'today', 'overall'.
    """
    logging.debug("[get_detection_analytics] Analytics endpoint hit")

    ist = pytz_timezone('Asia/Kolkata')
    time_range = request.args.get('range', 'overall')
    logging.debug(f"[get_detection_analytics] Time range received: {time_range}")

    now_utc = datetime.utcnow().replace(tzinfo=utc)

    # Define time deltas for short ranges
    time_deltas = {
        '5min': timedelta(minutes=5),
        '30min': timedelta(minutes=30),
        '1hr': timedelta(hours=1)
    }

    if time_range == 'today':
        now_ist = now_utc.astimezone(ist)
        since_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        since = since_ist.astimezone(utc)
        delta = now_utc - since
        logging.debug(f"[get_detection_analytics] Calculated 'today' range: since={since}, delta={delta}")
    elif time_range == 'overall':
        since = None
        delta = None
        logging.debug("[get_detection_analytics] Using 'overall' range, no time filtering applied.")
    else:
        delta = time_deltas.get(time_range, timedelta(minutes=5))  # fallback to 5m
        since = now_utc - delta
        logging.debug(f"[get_detection_analytics] Calculated short range: since={since}, delta={delta}")

    # Fetch logs within time window
    if time_range == 'overall':
        logging.debug("[get_detection_analytics] Fetching all detection logs (no time filter).")
        logs = DetectionLog.query.order_by(DetectionLog.timestamp.asc()).all()
    else:
        logging.debug(f"[get_detection_analytics] Fetching logs since: {since.isoformat()}")
        logs = DetectionLog.query.filter(
            DetectionLog.timestamp >= since
        ).order_by(DetectionLog.timestamp.asc()).all()

    logging.debug(f"[get_detection_analytics] Number of logs fetched: {len(logs)}")

    # Aggregate emotion counts for the current period
    emotion_counts = defaultdict(int)
    for log in logs:
        emotion = log.emotion.upper()
        emotion_counts[emotion] += 1

    logging.debug(f"[get_detection_analytics] Emotion counts: {dict(emotion_counts)}")

    total = sum(emotion_counts.values())
    logging.debug(f"[get_detection_analytics] Total detections: {total}")

    pie_data = [{'name': emo, 'value': count} for emo, count in emotion_counts.items()]
    logging.debug(f"[get_detection_analytics] Pie chart data: {pie_data}")

    pie_percentage_data = [
        {'name': emo, 'value': count, 'percentage': round(count / total * 100, 2)}
        for emo, count in emotion_counts.items()
    ] if total > 0 else []
    logging.debug(f"[get_detection_analytics] Pie chart with percentage: {pie_percentage_data}")

    most_frequent_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    logging.debug(f"[get_detection_analytics] Most frequent emotion: {most_frequent_emotion}")

    # Timeline aggregation with fixed emotions only
    emotions = ['FEAR', 'ANGER', 'SADNESS', 'DISGUST']
    timeline_buckets = defaultdict(lambda: {emo: 0 for emo in emotions})

    def bucket_time(ts, range_type):
        """Create time buckets based on range type"""
        ts_ist = ts.replace(tzinfo=utc).astimezone(ist)

        if range_type == 'overall':
            # For overall, group by day
            return ts_ist.strftime('%Y-%m-%d')
        elif range_type == 'today':
            # For today, group by hour
            return ts_ist.strftime('%H:00')
        else:
            # For short ranges (5m, 30m, 1h), group by 5-minute intervals
            minute_bucket = (ts_ist.minute // 5) * 5
            return ts_ist.replace(minute=minute_bucket, second=0, microsecond=0).strftime('%H:%M')

    for log in logs:
        if not log.timestamp:
            continue
        bucket = bucket_time(log.timestamp, time_range)
        emo = log.emotion.upper()
        if emo in emotions:
            timeline_buckets[bucket][emo] += 1
            logging.debug(f"[Timeline Aggregation] +1 for {emo} in bucket '{bucket}'")

    timeline_data = [{'time': time_label, **timeline_buckets[time_label]} for time_label in sorted(timeline_buckets)]
    logging.debug(f"[Timeline Aggregation] Final timeline data: {timeline_data}")

    # Previous period for trend comparison (skip for overall)
    trend = {}
    if time_range != 'overall' and delta is not None:
        previous_since = since - delta
        previous_until = since

        logging.debug(
            f"[Trend] Comparing current period ({since} to now) with previous period ({previous_since} to {previous_until})")

        previous_logs = DetectionLog.query.filter(
            DetectionLog.timestamp >= previous_since,
            DetectionLog.timestamp < previous_until
        ).all()

        previous_emotion_counts = defaultdict(int)
        for log in previous_logs:
            emo = log.emotion.upper()
            previous_emotion_counts[emo] += 1

        logging.debug(f"[Trend] Previous period emotion counts: {dict(previous_emotion_counts)}")
        logging.debug(f"[Trend] Current period emotion counts: {dict(emotion_counts)}")

        for emo in emotions:
            current = emotion_counts.get(emo, 0)
            previous = previous_emotion_counts.get(emo, 0)
            if previous == 0:
                trend[emo] = 'increase' if current > 0 else 'no change'
            else:
                if current > previous:
                    trend[emo] = 'increase'
                elif current < previous:
                    trend[emo] = 'decrease'
                else:
                    trend[emo] = 'no change'
            logging.debug(f"[Trend] {emo}: current={current}, previous={previous} → {trend[emo]}")
    else:
        # For overall, we can't compare with previous period, so mark as no change
        for emo in emotions:
            trend[emo] = 'no change'
            logging.debug(f"[Trend] {emo}: overall range → no change")

    # Peak times per emotion
    peak_times = {}
    for emo in emotions:
        peak_time, counts = max(
            timeline_buckets.items(),
            key=lambda x: x[1].get(emo, 0),
            default=(None, {})
        )
        peak_count = counts.get(emo, 0)
        peak_times[emo] = peak_time
        logging.debug(f"[PeakTime] Emotion: {emo}, Peak Time: {peak_time}, Count: {peak_count}")

    # Average confidence (intensity) per emotion
    confidence_sums = defaultdict(float)
    confidence_counts = defaultdict(int)

    for log in logs:
        if log.confidence is None:
            continue
        emo = log.emotion.upper()
        confidence_sums[emo] += log.confidence
        confidence_counts[emo] += 1

    avg_intensity = []
    for emo in confidence_counts:
        avg = round(confidence_sums[emo] / confidence_counts[emo], 2)
        avg_intensity.append({"name": emo, "value": avg})
        logging.debug(f"[AvgConfidence] Emotion: {emo}, Average Confidence: {avg}")

    # Additional metadata for overall stats
    date_range = {}
    if logs:
        first_log = min(logs, key=lambda x: x.timestamp)
        last_log = max(logs, key=lambda x: x.timestamp)
        date_range = {
            'start_date': first_log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': last_log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'total_days': (last_log.timestamp - first_log.timestamp).days + 1
        }
        logging.debug(f"[Analytics] Date range calculated: {date_range}")

    response_data = {
        'pie_data': pie_data,
        'pie_percentage_data': pie_percentage_data,
        'most_frequent_emotion': most_frequent_emotion,
        'timeline_data': timeline_data,
        'emotion_trends': trend,
        'peak_times': peak_times,
        'total_detections': total,
        'avg_intensity': avg_intensity,
        'time_range': time_range
    }

    logging.info(
        f"[Analytics] Time range: {time_range}, Total detections: {total}, Most frequent emotion: {most_frequent_emotion}")

    # Add date range info for overall view
    if time_range == 'overall':
        response_data['date_range'] = date_range
        logging.debug(f"[Analytics] Included date_range in response: {date_range}")

    logging.info(f"[Analytics] Final response prepared for time_range={time_range} with {len(logs)} logs.")

    return jsonify(response_data)


@camera_bp.route('/api/cameras/<int:camera_id>', methods=['GET'])
def get_camera(camera_id):
    logging.info(f"[Camera API] Fetching details for camera_id={camera_id}")

    camera = Camera.query.get(camera_id)
    if not camera:
        logging.warning(f"[Camera API] Camera with ID {camera_id} not found")
        return jsonify({'error': 'Camera not found'}), 404

    logging.debug(f"[Camera API] Camera found: {camera.to_dict()}")
    return jsonify(camera.to_dict()), 200


@camera_bp.route('/api/cameras/add', methods=['POST'])
def create_camera():
    logging.info("[Camera API] Received request to add a new camera.")
    data = request.json
    required_fields = ['label', 'ip', 'src']

    if not all(field in data for field in required_fields):
        logging.warning("[Camera API] Missing required fields in request payload.")
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        # Save to DB
        camera = Camera(
            label=data['label'],
            ip=data['ip'],
            src=data['src'],
            status=CameraStatus[data.get('status', 'Inactive')]
        )
        db.session.add(camera)
        db.session.commit()

        logging.info(f"[Camera API] Camera added successfully: {camera.to_dict()}")
        return jsonify(camera.to_dict()), 201

    except Exception as e:
        db.session.rollback()
        logging.error(f"[Camera API] Error while adding camera: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400


@camera_bp.route('/api/cameras/delete/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    logging.info(f"[Camera API] Received request to delete camera ID {camera_id}")
    camera = Camera.query.get(camera_id)
    if not camera:
        logging.warning(f"[Camera API] Camera ID {camera_id} not found")
        return jsonify({'error': 'Camera not found'}), 404

    # Notify emotion detection service to stop the camera detector
    try:
        response = requests.post(
            f'http://{ipaddress}/camera_status_update',
            json={'camera_id': camera_id, 'status': 'Inactive'},
            timeout=3
        )
        if response.status_code != 200:
            logging.warning(f"[Camera API] Failed to notify emotion detection service: {response.text}")
    except Exception as e:
        logging.error(f"[Camera API] Could not reach emotion detection service: {e}", exc_info=True)
        raise CustomException(e, sys)

    db.session.delete(camera)
    db.session.commit()
    logging.info(f"[Camera API] Camera ID {camera_id} deleted successfully")

    return jsonify({'message': 'Camera deleted successfully'}), 200


@camera_bp.route('/<int:camera_id>/status', methods=['PATCH'])
def update_camera_status(camera_id):
    logging.info(f"[Camera API] PATCH request to update status for camera ID {camera_id}")

    camera = Camera.query.get(camera_id)
    if not camera:
        logging.warning(f"[Camera API] Camera ID {camera_id} not found")
        return jsonify({'error': 'Camera not found'}), 404

    data = request.json
    try:
        camera.status = CameraStatus[data['status']]
        logging.info(f"[Camera API] Camera ID {camera_id} status set to {camera.status.value}")
    except KeyError:
        logging.error(f"[Camera API] Invalid status value received: {data.get('status')}")
        return jsonify({'error': 'Invalid status'}), 400

    db.session.commit()
    logging.info(f"[Camera API] Camera ID {camera_id} status updated in database")

    return jsonify({'message': f'Status updated to {camera.status.value}'}), 200


@camera_bp.route('/api/cameras/update/<int:camera_id>', methods=['PUT'])
def update_camera(camera_id):
    from models import Camera, CameraStatus
    from extensions import db
    import requests
    import logging

    logging.info(f"[Camera Update] PUT request received for camera ID: {camera_id}")

    data = request.get_json()
    if not data:
        logging.warning("[Camera Update] No input data provided")
        return jsonify({'error': 'No input data provided'}), 400

    label = data.get('label')
    ip = data.get('ip')
    src = data.get('src')  # <-- new field
    status = data.get('status')

    # Validate required fields (including src)
    if not label or not ip or not src or not status:
        logging.warning("[Camera Update] Missing required fields: label, ip, src, or status")
        return jsonify({'error': 'label, ip, src, and status are required'}), 400

    valid_status_values = [e.value for e in CameraStatus]
    if status not in CameraStatus.__members__ and status not in valid_status_values:
        logging.error(f"[Camera Update] Invalid status '{status}' provided")
        return jsonify({'error': f'Invalid status value. Must be one of {valid_status_values}'}), 400

    camera = Camera.query.get(camera_id)
    if not camera:
        logging.warning(f"[Camera Update] Camera ID {camera_id} not found in DB")
        return jsonify({'error': 'Camera not found'}), 404

    # Convert status string to Enum
    try:
        camera.status = CameraStatus[status] if isinstance(status, str) else status
    except KeyError:
        try:
            camera.status = CameraStatus(status)
        except Exception as e:
            logging.error(f"[Camera Update] Failed to convert status: {e}")
            return jsonify({'error': 'Invalid status format'}), 400

    # Update camera fields
    camera.label = label
    camera.ip = ip
    camera.src = src

    try:
        db.session.commit()
        logging.info(f"[Camera Update] Camera {camera_id} updated in DB")
    except IntegrityError:
        db.session.rollback()
        logging.warning(f"[Camera Update] IntegrityError on update for camera {camera_id} (label/IP/src conflict)")
        return jsonify({'error': 'Label, IP, or src must be unique, conflict detected'}), 409
    except Exception as e:
        db.session.rollback()
        logging.error(f"[Camera Update] DB commit failed for camera {camera_id}: {e}")
        return jsonify({'error': 'Failed to update camera'}), 500

    # Notify emotion worker service
    try:
        notify_url = f"http://{ipaddress}:5001/camera_status_update"
        payload = {"camera_id": camera_id, "status": camera.status.name}
        r = requests.post(notify_url, json=payload)
        r.raise_for_status()
        logging.info(f"[Camera Update] Notified emotion_worker_service about camera {camera_id} status change")
    except requests.exceptions.RequestException as e:
        logging.warning(f"[Camera Update] Failed to notify emotion_worker_service: {e}")
        raise CustomException(e, sys)

    return jsonify(camera.to_dict()), 200


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)


def run_emotion_detection(input_path, output_path):
    logging.info(f"🚀 Starting emotion detection on {input_path}")

    device = torch.device("cpu")
    model = ResEmoteNet().to(device)

    try:
        checkpoint = torch.load('models/fer_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info("✅ Model loaded successfully")
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    emotions = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"❌ Failed to open video file: {input_path}")
        return

    frame_width, frame_height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0
    processed_faces = 0

    def detect_emotion(frame):
        try:
            frame_tensor = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(frame_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy().flatten()
        except Exception as e:
            logging.warning(f"⚠️ Emotion detection failed on frame: {e}")
            return np.zeros(len(emotions))

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("📽️ End of video reached")
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop = frame[y:y + h, x:x + w]
            scores = detect_emotion(crop)
            label_idx = np.argmax(scores)
            label = emotions[label_idx]
            conf = scores[label_idx]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label.upper()} ({conf:.2f})", (x, y - 10), font, 1.0, (0, 255, 0), 2)

            processed_faces += 1

        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            logging.info(f"📦 Processed {frame_count} frames, {processed_faces} faces so far")

    cap.release()
    out.release()
    logging.info(f"✅ Emotion detection completed. Output saved to {output_path}")


@camera_bp.route('/api/process_video', methods=['POST'])
def process_uploaded_video():
    file = request.files.get('video')
    if not file:
        logging.warning("❌ No file uploaded in request")
        return jsonify({'error': 'No file uploaded'}), 400

    filename = f"{uuid.uuid4().hex}.mp4"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        logging.info(f"📥 Video file saved to {filepath}")
    except Exception as e:
        logging.error(f"❌ Failed to save uploaded file: {e}")
        return jsonify({'error': 'Failed to save uploaded file'}), 500

    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")

    try:
        run_emotion_detection(filepath, output_path)
        logging.info(f"✅ Processing completed for {filepath}")
    except Exception as e:
        logging.error(f"❌ Emotion detection failed: {e}")
        return jsonify({'error': 'Processing failed'}), 500

    return jsonify({'filename': f"processed_{filename}"}), 200


MODEL_PATH = 'app/camera/fer_model.pth'
# Setup device, model, emotions and transforms just like before
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

emotions = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model = ResEmoteNet().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Haarcascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Text settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 255, 0)
thickness = 2
line_type = cv2.LINE_AA


def predict_emotion(cv2_img, model_path=None):
    logging.debug("Starting predict_emotion")

    # Convert OpenCV image (BGR) to PIL image (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    logging.debug("Converted OpenCV BGR image to PIL RGB image")

    scores = detect_emotion(pil_img)
    logging.debug(f"Emotion scores: {scores}")

    label_index = np.argmax(scores)
    label = emotions[label_index]
    confidence = float(scores[label_index])
    logging.info(f"Predicted emotion: {label} with confidence {confidence:.4f}")

    return label, confidence


def base64_to_cv2_img(base64_str):
    logging.debug("Starting base64_to_cv2_img")

    # Remove header if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
        logging.debug("Removed base64 header")

    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logging.warning("Failed to decode image from base64 string")
        else:
            logging.debug("Image decoded successfully from base64")
        return img
    except Exception as e:
        logging.error(f"Error decoding base64 to image: {e}")
        return None


def cv2_img_to_base64(cv2_img):
    logging.debug("Starting cv2_img_to_base64")
    try:
        success, buffer = cv2.imencode('.jpg', cv2_img)
        if not success:
            logging.warning("cv2.imencode failed to encode image")
            return None
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        logging.debug("Image successfully encoded to base64")
        return jpg_as_text
    except Exception as e:
        logging.error(f"Error encoding cv2 image to base64: {e}")
        return None


def detect_emotion(pil_img):
    logging.debug("Starting emotion detection")
    try:
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = f.softmax(output, dim=1)
        scores = probabilities.cpu().numpy().flatten()
        logging.debug(f"Emotion detection completed with scores: {scores}")
        return scores
    except Exception as e:
        logging.error(f"Error during emotion detection: {e}")
        return None


@camera_bp.route('/api/emotion-detect', methods=['POST'])
def emotion_detect():
    try:
        data = request.json
        img_b64 = data.get('image')
        if not img_b64:
            logging.warning("No image provided in request")
            return jsonify({"error": "No image provided"}), 400

        img = base64_to_cv2_img(img_b64)
        if img is None:
            logging.warning("Invalid image data received")
            return jsonify({"error": "Invalid image data"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            logging.info("No face detected in the provided image")
            return jsonify({"error": "No face detected"}), 200

        x, y, w, h = faces[0]
        face_img = img[y:y + h, x:x + w]
        label, confidence = predict_emotion(face_img)

        logging.info(f"Detected emotion: {label} with confidence {confidence:.2f}")

        return jsonify({
            "label": label,
            "confidence": confidence,
            "face": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        })
    except Exception as e:
        logging.error(f"Error in emotion_detect: {e}", exc_info=True)
        return jsonify({"error": "Server error"}), 500


@camera_bp.route('/api/detection-logs/<int:log_id>', methods=['DELETE'])
def delete_detection_log(log_id):
    """Delete a specific detection log by ID"""
    try:
        log = DetectionLog.query.get(log_id)

        if not log:
            logging.warning(f"Detection log with id {log_id} not found for deletion")
            return jsonify({
                'success': False,
                'message': 'Detection log not found'
            }), 404

        image_path = log.image_path

        db.session.delete(log)
        db.session.commit()
        logging.info(f"Detection log {log_id} deleted from database")

        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logging.info(f"Deleted associated image file: {image_path}")
            except OSError as e:
                logging.warning(f"Could not delete image file {image_path}: {e}")
                raise CustomException(e, sys)

        return jsonify({
            'success': True,
            'message': 'Detection log deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting detection log {log_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error deleting detection log: {str(e)}'
        }), 500


@camera_bp.route('/api/detection-logs/bulk', methods=['DELETE'])
def bulk_delete_detection_logs():
    """Delete multiple detection logs by IDs"""
    try:
        data = request.get_json()
        if not data or 'ids' not in data:
            logging.warning("Bulk delete called without 'ids' in request")
            return jsonify({
                'success': False,
                'message': 'No log IDs provided'
            }), 400

        log_ids = data['ids']
        if not isinstance(log_ids, list) or not log_ids:
            logging.warning(f"Invalid log IDs format received: {log_ids}")
            return jsonify({
                'success': False,
                'message': 'Invalid log IDs format'
            }), 400

        logs = DetectionLog.query.filter(DetectionLog.id.in_(log_ids)).all()

        if not logs:
            logging.warning(f"No detection logs found for IDs: {log_ids}")
            return jsonify({
                'success': False,
                'message': 'No logs found with provided IDs'
            }), 404

        image_paths = [log.image_path for log in logs if log.image_path]

        deleted_count = 0
        for log in logs:
            db.session.delete(log)
            deleted_count += 1

        db.session.commit()
        logging.info(f"Deleted {deleted_count} detection logs from database")

        deleted_images = 0
        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    deleted_images += 1
                    logging.info(f"Deleted image file: {image_path}")
                except OSError as e:
                    logging.warning(f"Could not delete image file {image_path}: {e}")
                    raise CustomException(e, sys)

        return jsonify({
            'success': True,
            'message': f'Successfully deleted {deleted_count} detection logs and {deleted_images} image files'
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting detection logs in bulk: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error deleting detection logs: {str(e)}'
        }), 500


@camera_bp.route('/api/detection-logs/camera/<int:camera_id>', methods=['DELETE'])
def delete_logs_by_camera(camera_id):
    """Delete all detection logs for a specific camera"""
    try:
        logs = DetectionLog.query.filter_by(camera_id=camera_id).all()

        if not logs:
            logging.info(f"No detection logs found for camera ID {camera_id}")
            return jsonify({
                'success': True,
                'message': f'No logs found for camera ID {camera_id}'
            }), 200

        image_paths = [log.image_path for log in logs if log.image_path]

        deleted_count = DetectionLog.query.filter_by(camera_id=camera_id).delete()
        db.session.commit()
        logging.info(f"Deleted {deleted_count} detection logs for camera ID {camera_id} from database")

        deleted_images = 0
        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    deleted_images += 1
                    logging.info(f"Deleted image file: {image_path}")
                except OSError as e:
                    logging.warning(f"Could not delete image file {image_path}: {e}")
                    raise CustomException(e, sys)

        return jsonify({
            'success': True,
            'message': f'Successfully deleted {deleted_count} detection logs for camera {camera_id} and {deleted_images} image files'
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting logs for camera {camera_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error deleting logs for camera {camera_id}: {str(e)}'
        }), 500


@camera_bp.route('/api/detection-logs/clear-all', methods=['DELETE'])
def clear_all_detection_logs():
    """Delete ALL detection logs (use with caution)"""
    try:
        confirm = request.args.get('confirm', '').lower()
        if confirm != 'true':
            logging.warning("Clear all detection logs called without confirmation")
            return jsonify({
                'success': False,
                'message': 'This action requires confirmation. Add ?confirm=true to the URL'
            }), 400

        logs = DetectionLog.query.all()
        image_paths = [log.image_path for log in logs if log.image_path]

        deleted_count = DetectionLog.query.delete()
        db.session.commit()
        logging.info(f"Deleted all detection logs: {deleted_count} records removed")

        deleted_images = 0
        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    deleted_images += 1
                    logging.info(f"Deleted image file: {image_path}")
                except OSError as e:
                    logging.warning(f"Could not delete image file {image_path}: {e}")
                    raise CustomException(e, sys)

        return jsonify({
            'success': True,
            'message': f'Successfully deleted all {deleted_count} detection logs and {deleted_images} image files'
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error clearing all detection logs: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error clearing all detection logs: {str(e)}'
        }), 500


@camera_bp.route('/api/detection-logs/cleanup-old', methods=['DELETE'])
def cleanup_old_detection_logs():
    """Delete detection logs older than specified days"""
    try:
        days = request.args.get('days', 30, type=int)

        if days <= 0:
            logging.warning(f"Cleanup called with non-positive days parameter: {days}")
            return jsonify({
                'success': False,
                'message': 'Days parameter must be positive'
            }), 400

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        logging.info(f"Cleaning up detection logs older than {days} days (before {cutoff_date})")

        old_logs = DetectionLog.query.filter(DetectionLog.timestamp < cutoff_date).all()

        if not old_logs:
            logging.info(f"No detection logs found older than {days} days")
            return jsonify({
                'success': True,
                'message': f'No logs older than {days} days found'
            }), 200

        image_paths = [log.image_path for log in old_logs if log.image_path]

        deleted_count = DetectionLog.query.filter(DetectionLog.timestamp < cutoff_date).delete()
        db.session.commit()
        logging.info(f"Deleted {deleted_count} detection logs older than {days} days")

        deleted_images = 0
        for image_path in image_paths:
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    deleted_images += 1
                    logging.info(f"Deleted image file: {image_path}")
                except OSError as e:
                    logging.warning(f"Could not delete image file {image_path}: {e}")
                    raise CustomException(e, sys)

        return jsonify({
            'success': True,
            'message': f'Successfully deleted {deleted_count} logs older than {days} days and {deleted_images} image files'
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error cleaning up old detection logs: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error cleaning up old detection logs: {str(e)}'
        }), 500
