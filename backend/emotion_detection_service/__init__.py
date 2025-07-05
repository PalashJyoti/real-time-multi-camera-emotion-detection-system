import sys

from flask import Flask

from emotion_detection_service.exception import CustomException
from emotion_detection_service.logger import logging
from extensions import db  # assuming extensions.py is in the project root or accessible


class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'  # adjust path as needed
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'your_secret_key'  # Optional for sessions, CSRF, etc.


def create_app():
    logging.debug("Starting app creation...")

    app = Flask(__name__)
    app.config.from_object(Config)
    logging.debug("App configuration loaded.")

    # Initialize extensions
    db.init_app(app)
    logging.debug("Database initialized.")

    try:
        from emotion_detection_service.routes import api_bp
        app.register_blueprint(api_bp)
        logging.debug("Blueprint 'api_bp' registered.")
    except Exception as e:
        logging.error(f"Failed to register blueprint: {e}")
        raise CustomException(e, sys)

    logging.info("App creation completed.")
    return app
