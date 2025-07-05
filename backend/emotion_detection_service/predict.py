import cv2
import numpy as np
import torch
import torch.nn.functional as F

from emotion_detection_service.globals import load_model, _model, _device, _labels, _transform
from emotion_detection_service.logger import logging

# Flag to avoid reloading model on every call
_model_loaded = False


def predict_emotion(frame, model_path: str):
    global _model_loaded

    try:
        # Load model only once
        if not _model_loaded:
            load_model(model_path)
            _model_loaded = True

        if _model is None:
            logging.error("Model is not loaded properly.")
            return "error", 0.0

        # Convert image to RGB and preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = _transform(rgb).unsqueeze(0).to(_device)

        with torch.no_grad():
            logits = _model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        return _labels[idx], float(probs[idx])

    except Exception as e:
        logging.error(f"[predict_emotion] Error: {e}")
        return "error", 0.0


# Add batch processing capability
def predict_emotions_batch(frames, model_path: str):
    global _model_loaded

    try:
        # Load model only once
        if not _model_loaded:
            load_model(model_path)
            _model_loaded = True

        if _model is None:
            logging.error("Model is not loaded properly.")
            return [("error", 0.0)] * len(frames)

        # Process batch of images
        batch_tensors = []
        for frame in frames:
            # Convert image to RGB and preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = _transform(rgb).unsqueeze(0)
            batch_tensors.append(img_tensor)

        # Stack tensors into a batch
        if batch_tensors:
            batch = torch.cat(batch_tensors, 0).to(_device)

            with torch.no_grad():
                logits = _model(batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()

            results = []
            for i in range(len(frames)):
                idx = int(np.argmax(probs[i]))
                results.append((_labels[idx], float(probs[i][idx])))
            return results
        return []

    except Exception as e:
        logging.error(f"[predict_emotions_batch] Error: {e}")
        return [("error", 0.0)] * len(frames)
