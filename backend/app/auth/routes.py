import io
from datetime import datetime, timezone, timedelta
from functools import wraps

import jwt
import pyotp
import qrcode
from flask import Blueprint, request, jsonify, send_file, g
from pytz import timezone as pytz_timezone, utc

from extensions import db
from logger import logging
from models import User

auth_bp = Blueprint('auth', __name__)
JWT_SECRET = 'your_jwt_secret_key'  # Replace this with a secure env var in production
JWT_EXP_DELTA_SECONDS = 3600

blacklisted_tokens = set()


def create_jwt(user):
    payload = {
        'user_id': user.id,
        'username': user.username,
        'role': user.role,
        'exp': datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')


def decode_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError as e:
        logging.warning(f"JWT token expired: {e}")
        return None
    except jwt.InvalidTokenError as e:
        logging.warning(f"Invalid JWT token: {e}")
        return None


def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        logging.info(f"Accessing protected route: {request.path}")

        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logging.warning("JWT auth failed: Authorization header missing")
            return jsonify({'error': 'Authorization header missing'}), 401

        token = auth_header.replace('Bearer ', '').strip()
        if not token:
            logging.warning("JWT auth failed: Token missing in header")
            return jsonify({'error': 'Token missing'}), 401

        if token in blacklisted_tokens:
            logging.warning("JWT auth failed: Token is blacklisted")
            return jsonify({'error': 'Token is blacklisted'}), 401

        payload = decode_jwt(token)
        if not payload:
            logging.warning("JWT auth failed: Invalid or expired token")
            return jsonify({'error': 'Invalid or expired token'}), 401

        g.user = payload
        logging.info(f"JWT auth success for user: {payload.get('username')} (Role: {payload.get('role')})")

        return f(*args, **kwargs)

    return decorated


@auth_bp.route('/signup', methods=['POST'])
def signup():
    logging.info("Signup endpoint hit")

    data = request.json
    username = data.get('username')
    name = data.get('name')
    password = data.get('password')

    if not username or not name or not password:
        logging.warning("Signup failed: Missing required fields")
        return jsonify({'error': 'All fields are required'}), 400

    if User.query.filter_by(username=username).first():
        logging.warning(f"Signup failed: User '{username}' already exists")
        return jsonify({'error': 'User already exists'}), 409

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    otp_url = totp.provisioning_uri(name=username, issuer_name="MindSightAI")
    logging.debug(f"TOTP provisioning URI created for user '{username}'")

    img = qrcode.make(otp_url)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    user = User(name=name, username=username, secret=secret, role='user')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    logging.info(f"User '{username}' signed up successfully")

    return send_file(buf, mimetype='image/png')


@auth_bp.route('/login', methods=['POST'])
def login():
    logging.info("Login endpoint hit")

    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        logging.warning(f"Login failed for user '{username}': Invalid credentials")
        return jsonify({'error': 'Invalid credentials'}), 401

    user.last_login = datetime.utcnow().replace(tzinfo=timezone.utc)
    db.session.commit()
    logging.info(f"User '{username}' authenticated successfully, TOTP required")

    return jsonify({'message': '2FA required'}), 200


@auth_bp.route('/verify-totp', methods=['POST'])
def verify_totp():
    logging.info("verify-totp endpoint hit")

    data = request.json
    username = data.get('username')
    token = data.get('token')

    if not username or not token:
        logging.warning("TOTP verification failed: Missing username or token")
        return jsonify({'error': 'Username and token are required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        logging.warning(f"TOTP verification failed: User '{username}' not found")
        return jsonify({'error': 'User not found'}), 404

    logging.debug(f"Verifying TOTP for user '{username}' with token: {token}")
    logging.debug(f"User secret: {user.secret}")

    totp = pyotp.TOTP(user.secret)
    if totp.verify(token):
        logging.info(f"TOTP verified successfully for user '{username}'")
        jwt_token = create_jwt(user)
        return jsonify({'token': jwt_token}), 200
    else:
        logging.warning(f"TOTP verification failed for user '{username}': Invalid token")
        return jsonify({'error': 'Invalid TOTP'}), 401


@auth_bp.route('/verify-totp-for-reset', methods=['POST'])
def verify_totp_for_reset():
    logging.info("verify-totp-for-reset endpoint hit")

    data = request.json
    username = data.get('username')
    token = data.get('token')

    user = User.query.filter_by(username=username).first()
    if not user:
        logging.warning(f"TOTP reset verification failed: User '{username}' not found")
        return jsonify({'error': 'User not found'}), 404

    logging.debug(f"Verifying TOTP for password reset for user '{username}' with token: {token}")

    totp = pyotp.TOTP(user.secret)
    if totp.verify(token):
        logging.info(f"TOTP for password reset verified successfully for user '{username}'")
        return jsonify({'message': 'TOTP verified successfully'}), 200
    else:
        logging.warning(f"TOTP reset verification failed for user '{username}': Invalid token")
        return jsonify({'error': 'Invalid token'}), 401


@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    logging.info("reset-password endpoint hit")

    data = request.json
    username = data.get('username')
    new_password = data.get('newPassword')

    if not new_password:
        logging.warning(f"Password reset failed: New password missing for user '{username}'")
        return jsonify({'error': 'New password is required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        logging.warning(f"Password reset failed: User '{username}' not found")
        return jsonify({'error': 'User not found'}), 404

    user.set_password(new_password)
    db.session.commit()
    logging.info(f"Password reset successful for user '{username}'")

    return jsonify({'message': 'Password reset successfully'}), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required
def logout():
    logging.info("logout endpoint hit")

    auth_header = request.headers.get('Authorization')
    if not auth_header:
        logging.warning("Logout failed: Authorization header missing")
        return jsonify({'error': 'Authorization header missing'}), 401

    token = auth_header.replace('Bearer ', '')
    blacklisted_tokens.add(token)
    logging.info("User logged out successfully and token blacklisted")

    return jsonify({'message': 'Logged out successfully'}), 200


@auth_bp.route('/change-role', methods=['POST'])
@jwt_required
def change_role():
    logging.info("change-role endpoint hit")

    current_user = User.query.get(g.user['user_id'])
    if current_user.role != 'admin':
        logging.warning(f"Role change denied: User '{current_user.username}' is not an admin")
        return jsonify({'error': 'Access denied. Admins only.'}), 403

    data = request.json
    username = data.get('username')
    new_role = data.get('role')

    if not username or not new_role:
        logging.warning("Role change failed: Missing username or role in request")
        return jsonify({'error': 'Username and role are required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        logging.warning(f"Role change failed: Target user '{username}' not found")
        return jsonify({'error': 'User not found'}), 404

    user.role = new_role
    db.session.commit()
    logging.info(f"Role for user '{username}' changed to '{new_role}' by admin '{current_user.username}'")

    return jsonify({'message': f"Role for {username} changed to {new_role}"}), 200


@auth_bp.route('/users', methods=['GET'])
def get_users():
    logging.info(f"get-users endpoint hit by user '{g.user.get('username')}'")

    ist = pytz_timezone('Asia/Kolkata')
    users = User.query.all()

    logging.debug(f"Fetched {len(users)} users from database")

    return jsonify({
        'users': [
            {
                'id': user.id,
                'name': user.name,
                'role': user.role,
                'last_login': user.last_login.replace(tzinfo=utc).astimezone(ist).strftime(
                    '%b %d, %Y, %I:%M %p') if user.last_login else None
            }
            for user in users
        ]
    })


@auth_bp.route('/users/add', methods=['POST'])
def add_user():
    logging.info("add-user endpoint hit")

    data = request.json
    name = data.get('name')
    role = data.get('role', 'user')

    if not name:
        logging.warning("Add user failed: 'name' field is missing")
        return jsonify({'error': 'Name is required'}), 400

    user = User(name=name, role=role)
    db.session.add(user)
    db.session.commit()

    logging.info(f"User '{name}' added with role '{role}'")
    return jsonify({'message': 'User added'}), 201


@auth_bp.route('/users/delete/<int:user_id>', methods=['DELETE'])
@jwt_required
def delete_user(user_id):
    logging.info(f"delete-user endpoint hit by user '{g.user.get('username')}' to delete user ID {user_id}")

    current_user = User.query.get(g.user['user_id'])
    if current_user.role != 'admin':
        logging.warning(f"Delete user denied: User '{current_user.username}' is not an admin")
        return jsonify({'error': 'Access denied. Admins only.'}), 403

    user = User.query.get(user_id)
    if not user:
        logging.warning(f"Delete user failed: Target user with ID {user_id} not found")
        return jsonify({'error': 'User not found'}), 404

    db.session.delete(user)
    db.session.commit()

    logging.info(f"User with ID {user_id} deleted by admin '{current_user.username}'")
    return jsonify({'message': 'User deleted'}), 200


@auth_bp.route('/users/<int:user_id>/role', methods=['PUT'])
@jwt_required
def update_role(user_id):
    logging.info(f"update-role endpoint hit by user '{g.user.get('username')}' to update user ID {user_id}")

    current_user = g.user

    if current_user['role'] != 'admin':
        logging.warning(f"Role update denied: User '{current_user['username']}' is not an admin")
        return jsonify({'error': 'Access denied. Admins only.'}), 403

    if user_id == current_user['user_id']:
        logging.warning(f"Role update denied: Admin '{current_user['username']}' attempted to change their own role")
        return jsonify({'error': 'You cannot change your own role.'}), 403

    user = User.query.get(user_id)
    if not user:
        logging.warning(f"Role update failed: Target user with ID {user_id} not found")
        return jsonify({'error': 'User not found'}), 404

    data = request.json
    new_role = data.get('role')
    if not new_role:
        logging.warning(f"Role update failed: New role not provided in request by '{current_user['username']}'")
        return jsonify({'error': 'New role is required'}), 400

    if new_role not in ['admin', 'user']:
        logging.warning(f"Role update failed: Invalid role '{new_role}' specified by '{current_user['username']}'")
        return jsonify({'error': 'Invalid role specified'}), 400

    user.role = new_role
    db.session.commit()

    logging.info(f"Role for user ID {user_id} changed to '{new_role}' by admin '{current_user['username']}'")

    return jsonify({'id': user.id, 'name': user.name, 'role': user.role})
