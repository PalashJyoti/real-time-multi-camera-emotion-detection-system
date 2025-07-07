import threading
import time

import cv2
from flask import Blueprint, Response, jsonify, current_app, request

import emotion_detection_service.globals as globals_module
from emotion_detection_service.logger import logging

emotion_bp = Blueprint('emotion', __name__)

# Use a thread-local storage for request context
local_storage = threading.local()


@emotion_bp.route('/stream/<int:camera_id>')
def video_feed(camera_id):
    def generate_frames():
        while True:
            frame = globals_module.manager.get_frame(camera_id)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # Return a blank frame or error image if no frame is available
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


from extensions import db

from models import Camera

api_bp = Blueprint('api', __name__)


@api_bp.route('/stream/<int:cam_id>', methods=['GET'])
def stream(cam_id):
    logging.info(f"📡 Incoming stream request for camera {cam_id}")

    def generate():
        while True:
            try:
                manager = globals_module.manager
                if manager is None:
                    logging.warning(f"⚠️ Camera manager is None while streaming cam_id={cam_id}")
                    time.sleep(0.1)
                    continue

                frame = manager.get_frame(cam_id)
                if frame is not None:
                    _, buf = cv2.imencode('.jpg', frame)
                    try:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                    except GeneratorExit:
                        logging.info(f"Client disconnected from cam_id={cam_id} stream")
                        break
                    except Exception as e:
                        logging.error(f"Error yielding frame for cam_id={cam_id}: {e}")
                        break
                else:
                    logging.debug(f"🕳️ No frame available for cam_id={cam_id}")
                    time.sleep(0.05)

            except Exception as e:
                logging.error(f"❌ Exception in stream generator for cam_id={cam_id}: {e}", exc_info=True)
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@api_bp.route('/camera_status_update', methods=['POST'])
def camera_status_update():
    data = request.get_json()
    camera_id = data.get('camera_id')
    status = data.get('status')

    if camera_id is None or status is None:
        logging.warning("⚠️ Missing camera_id or status in POST data")
        return jsonify({'error': 'camera_id and status are required'}), 400

    logging.info(f"🔄 Camera status update received - ID: {camera_id}, Status: {status}")

    try:
        with current_app.app_context():
            if status == 'Inactive':
                globals_module.manager.remove_camera(camera_id)
                logging.info(f"📴 Camera {camera_id} set to Inactive and removed from manager")

            elif status == 'Active':
                cam = db.session.get(Camera, camera_id)
                if not cam:
                    logging.error(f"❌ Camera with ID {camera_id} not found in DB")
                    return jsonify({'error': 'Camera not found'}), 404
                globals_module.manager.add_camera(cam.id, cam.src)
                logging.info(f"📶 Camera {cam.id} activated with source: {cam.src}")

            else:
                logging.warning(f"⚠️ Invalid status value provided: {status}")
                return jsonify({'error': 'Invalid status'}), 400

        return jsonify({'message': f'Status updated to {status}'}), 200

    except Exception as e:
        logging.exception(f"❌ Error in camera_status_update for cam_id={camera_id}")
        return jsonify({'error': str(e)}), 500
