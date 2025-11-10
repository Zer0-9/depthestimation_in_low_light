import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt


# -------------------------
# Model Definitions
# -------------------------
class DepthDecoder(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


class DepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        modules = list(backbone.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        self.decoder = DepthDecoder(2048)

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return F.softplus(out)


# -------------------------
# Load Model
# -------------------------
def load_model(weights_path, device):
    model = DepthModel().to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"[INFO] Loaded model from: {weights_path}")
    return model


# -------------------------
# Depth Prediction
# -------------------------
def predict_depth(model, img_path, device, img_size=(384, 384)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_t)[0, 0].cpu().numpy()

    # -------------------------
    # Normalize Depth (remove outliers)
    # -------------------------
    min_val, max_val = np.percentile(pred, (2, 98))  # Ignore outliers
    pred_clipped = np.clip(pred, min_val, max_val)
    depth_norm = (pred_clipped - min_val) / (max_val - min_val + 1e-8)

    # Generate visualization maps
    depth_gray = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_VIRIDIS)

    return pred, depth_gray, depth_color


# -------------------------
# Main Inference
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "results_depthdark/best_model.pth"
    test_images = ["test3.jpg"]
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, device)

    for img_path in test_images:
        print(f"[INFO] Processing: {img_path}")
        depth_raw, depth_gray, depth_color = predict_depth(model, img_path, device)

        stem = Path(img_path).stem
        np.save(os.path.join(output_dir, f"{stem}_depth.npy"), depth_raw)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_depth_gray.png"), depth_gray)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_depth_color.png"), depth_color)
        print(f"[INFO] Saved: {stem}_depth_color.png")

        # Visualization
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.title("Input RGB")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(depth_gray, cmap="gray")
        plt.title("Depth (Grayscale)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
        plt.title("Depth (Viridis)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
