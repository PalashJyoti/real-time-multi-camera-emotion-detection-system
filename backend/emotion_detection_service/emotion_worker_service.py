import sys
import threading
import time

import emotion_detection_service.globals as globals_module
from emotion_detection_service import create_app
from emotion_detection_service.exception import CustomException
from emotion_detection_service.globals import init_camera_manager, load_model
from emotion_detection_service.logger import logging
from models import Camera, CameraStatus

app = create_app()
model_path = "emotion_detection_service/fer_model.pth"

# Module-level variable
cleanup_thread_started = False


def start_cleanup_loop(camera_manager, interval=30):
    global cleanup_thread_started
    if cleanup_thread_started:
        logging.warning("Cleanup loop already started.")
        return None  # Or return existing thread if you're keeping it

    def cleanup_loop():
        logging.info("Starting cleanup loop thread.")
        while True:
            time.sleep(interval)
            try:
                camera_manager.cleanup_inactive_cameras()
                logging.debug("Called cleanup method")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
                raise CustomException(e, sys)

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    cleanup_thread_started = True
    return thread


def run_flask():
    logging.info("Starting Flask server on 0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True)


if __name__ == "__main__":
    logging.info("Loading model...")
    try:
        load_model(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    globals_module.manager = init_camera_manager(model_path=model_path, app=app)
    cleanup_thread = start_cleanup_loop(globals_module.manager)

    with app.app_context():
        active_cameras = Camera.query.filter_by(status=CameraStatus.Active).all()
        for cam in active_cameras:
            logging.info(f"Adding active camera: ID={cam.id}, SRC={cam.src}")
            globals_module.manager.add_camera(cam.id, cam.src)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logging.info("Emotion detection & streaming service started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt as e:
        logging.info("Stopping all detectors...")
        globals_module.manager.stop_all()
        logging.info("Shutdown complete.")
        raise CustomException(e, sys)
