import sys
from threading import Lock
from typing import Optional

import torch
from torchvision import transforms

from emotion_detection_service.exception import CustomException
from emotion_detection_service.logger import logging
from emotion_detection_service.model import ResEmoteNet

# Face detector model files (adjust if needed)
FACE_DETECTOR_PROTO = "emotion_detection_service/deploy.prototxt"
FACE_DETECTOR_MODEL = "emotion_detection_service/res10_300x300_ssd_iter_140000.caffemodel"

# Internal globals
manager = None
_model: Optional[ResEmoteNet] = None
_model_lock = Lock()
_labels = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # standard ImageNet
])


def load_model(model_path: str):
    global _model
    with _model_lock:
        if _model is None:
            logging.debug("Loading emotion detection model...")
            try:
                _model = ResEmoteNet().to(_device)
                checkpoint = torch.load(model_path, map_location=_device)
                _model.load_state_dict(checkpoint['model_state_dict'])
                _model.eval()
                logging.info("Emotion detection model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load model: {e}")
                raise CustomException(e, sys)


def init_camera_manager(model_path: str, app):
    global manager
    from emotion_detection_service.multi_camera_manager import MultiCameraManager
    logging.debug("Initializing multi-camera manager...")
    manager = MultiCameraManager(model_path=model_path, app=app)
    return manager


# Enhanced GPU detection and configuration
def get_optimal_device():
    if torch.cuda.is_available():
        # Get the GPU with the most free memory
        device_count = torch.cuda.device_count()
        if device_count > 1:
            free_mem = []
            for i in range(device_count):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_mem.append(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i))
            device_id = free_mem.index(max(free_mem))
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Apple Silicon (M1/M2/M3) Macs
        return torch.device('mps')
    else:
        return torch.device('cpu')


# Use the optimal device
_device = get_optimal_device()
logging.info(f"Using device: {_device}")

# Configure PyTorch for better performance
if _device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    logging.info(f"CUDA Device: {torch.cuda.get_device_name(_device.index)}")
    logging.info(f"CUDA Capability: {torch.cuda.get_device_capability(_device.index)}")
