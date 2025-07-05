import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import logging


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        logging.debug(f"[SEBlock] Initialized with in_channels={in_channels}, reduction={reduction}")

    def forward(self, x):
        b, c, _, _ = x.size()
        logging.debug(f"[SEBlock] Forward pass input shape: {x.shape}")

        y = self.avg_pool(x).view(b, c)
        logging.debug(f"[SEBlock] After avg_pool and view: {y.shape}")

        y = self.fc(y).view(b, c, 1, 1)
        logging.debug(f"[SEBlock] After FC and view: {y.shape}")

        out = x * y.expand_as(x)
        logging.debug(f"[SEBlock] Output shape: {out.shape}")
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_ch)
            )

        logging.debug(f"[ResidualBlock] Initialized with in_ch={in_ch}, out_ch={out_ch}, stride={stride}")

    def forward(self, x):
        logging.debug(f"[ResidualBlock] Input shape: {x.shape}")
        out = F.relu(self.bn1(self.conv1(x)))
        logging.debug(f"[ResidualBlock] After conv1+bn1+relu: {out.shape}")

        out = self.bn2(self.conv2(out))
        logging.debug(f"[ResidualBlock] After conv2+bn2: {out.shape}")

        shortcut_out = self.shortcut(x)
        logging.debug(f"[ResidualBlock] Shortcut output shape: {shortcut_out.shape}")

        out += shortcut_out
        out = F.relu(out)
        logging.debug(f"[ResidualBlock] Output shape after addition and relu: {out.shape}")
        return out


class ResEmoteNet(nn.Module):
    def __init__(self):
        super(ResEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(256)

        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 7)

        logging.debug("[ResEmoteNet] Model initialized.")

    def forward(self, x):
        logging.debug(f"[ResEmoteNet] Input shape: {x.shape}")
        x = F.relu(self.bn1(self.conv1(x)))
        logging.debug(f"[ResEmoteNet] After conv1 -> relu: {x.shape}")
        x = F.max_pool2d(x, 2)
        logging.debug(f"[ResEmoteNet] After maxpool1: {x.shape}")
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        logging.debug(f"[ResEmoteNet] After conv2 -> relu: {x.shape}")
        x = F.max_pool2d(x, 2)
        logging.debug(f"[ResEmoteNet] After maxpool2: {x.shape}")
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        logging.debug(f"[ResEmoteNet] After conv3 -> relu: {x.shape}")
        x = F.max_pool2d(x, 2)
        logging.debug(f"[ResEmoteNet] After maxpool3: {x.shape}")

        x = self.se(x)
        logging.debug(f"[ResEmoteNet] After SEBlock: {x.shape}")

        x = self.res_block1(x)
        logging.debug(f"[ResEmoteNet] After res_block1: {x.shape}")
        x = self.res_block2(x)
        logging.debug(f"[ResEmoteNet] After res_block2: {x.shape}")
        x = self.res_block3(x)
        logging.debug(f"[ResEmoteNet] After res_block3: {x.shape}")

        x = self.pool(x)
        logging.debug(f"[ResEmoteNet] After AdaptiveAvgPool2d: {x.shape}")
        x = x.view(x.size(0), -1)
        logging.debug(f"[ResEmoteNet] After flatten: {x.shape}")

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logging.debug(f"[ResEmoteNet] After fc1 + dropout: {x.shape}")

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logging.debug(f"[ResEmoteNet] After fc2 + dropout: {x.shape}")

        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        logging.debug(f"[ResEmoteNet] After fc3 + dropout: {x.shape}")

        x = self.fc4(x)
        logging.debug(f"[ResEmoteNet] Final output shape: {x.shape}")
        return x


# Model initialization and prediction functions
_model = None
_labels = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']


def _load_model(path):
    global _model
    if _model is None:
        logging.debug(f"[ModelLoader] Loading ResEmoteNet model from {path}")
        _model = ResEmoteNet()
        checkpoint = torch.load(path, map_location='cpu')
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()
        logging.debug("[ModelLoader] Model loaded and set to eval mode.")
    else:
        logging.debug("[ModelLoader] Model already loaded; skipping reload.")


def predict_emotion(frame, model_path):
    """
    frame: BGR image (NumPy array).
    Returns: (label, confidence)
    """
    logging.debug("[predict_emotion] Starting prediction.")
    _load_model(model_path)

    # Resize and convert BGR to RGB
    face = cv2.resize(frame, (64, 64))
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    logging.debug(f"[predict_emotion] Input tensor shape: {tensor.shape}")

    _model.eval()  # ensure eval mode before prediction
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).numpy()

    idx = np.argmax(probs)
    label = _labels[idx]
    confidence = float(probs[idx])

    logging.debug(f"[predict_emotion] Predicted: {label} with confidence: {confidence:.4f}")
    return label, confidence
