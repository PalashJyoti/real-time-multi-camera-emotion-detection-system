from flask import Flask
from flask_cors import CORS
import pyotp
from extensions import db
import os
import sys
from sqlalchemy import inspect
from models import User, Camera, CameraStatus
from ip import ipaddress

# Add parent directory to sys.path so we can import from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_app():
    # Get path to parent directory (backend/)
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    app = Flask(
        __name__,
        static_folder=os.path.join(backend_dir, 'static'),
        static_url_path='/static'
    )
    app.secret_key = 'your_secret_key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    CORS(app, origins=[f"http://{ipaddress}:3001","http://localhost:3001"], supports_credentials=True)

    from app.auth.routes import auth_bp
    from app.camera.routes import camera_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(camera_bp)

    with app.app_context():
        # Create all tables
        db.create_all()

        inspector = inspect(db.engine)
        print("📋 Tables:", inspector.get_table_names())

        # --- Create Permanent Admin (if not exists) ---
        admin_username = 'admin'
        admin_password = 'secureAdmin123'

        existing_admin = User.query.filter_by(username=admin_username).first()
        if not existing_admin:
            totp_secret = pyotp.random_base32()

            admin_user = User(
                username=admin_username,
                name='Super Admin',
                role='admin',
                secret=totp_secret
            )
            admin_user.set_password(admin_password)

            db.session.add(admin_user)
            db.session.commit()

            otp_uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
                name=admin_username,
                issuer_name="MindSightAI"
            )

            print("✅ Permanent admin created.")
            print(f"🔐 Scan this QR in Google Authenticator:\n{otp_uri}")
        else:
            print("✅ Permanent admin already exists.")

        # # --- Add test video camera (for development) ---
        # test_label = "Test Video Camera"
        # test_src = "app/camera/video.mp4"
        # test_ip = "127.0.0.1"  # dummy IP for required field
        #
        # existing_camera = Camera.query.filter_by(label=test_label).first()
        # if not existing_camera:
        #     test_camera = Camera(
        #         label=test_label,
        #         ip=test_ip,
        #         src=test_src,
        #         status=CameraStatus.Active
        #     )
        #     db.session.add(test_camera)
        #     db.session.commit()
        #     print(f"🎥 Test video camera added with src: {test_src}")
        # else:
        #     print("🎥 Test video camera already exists.")

        # --- Initialize camera_manager but do NOT start emotion detectors ---
        # from app.camera.camera_manager import init_camera_manager
        # init_camera_manager()  # Initializes camera_manager but no threads started here

    return app