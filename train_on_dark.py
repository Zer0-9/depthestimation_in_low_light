import os
import argparse
import random
import csv
from glob import glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
def compute_metrics_numpy(pred, gt, mask=None):
    if mask is None:
        mask = (gt > 0)
    if mask.sum() == 0:
        return None
    p = pred[mask]
    g = gt[mask]
    p = np.clip(p, 1e-3, 1e3)
    g = np.clip(g, 1e-3, 1e3)

    abs_rel = np.mean(np.abs(g - p) / g)
    rmse = np.sqrt(np.mean((g - p) ** 2))
    log10 = np.mean(np.abs(np.log10(g) - np.log10(p)))
    thresh = np.maximum(g / p, p / g)
    d1 = np.mean(thresh < 1.25)
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)
    silog = np.sqrt(np.mean((np.log(p / g)) ** 2) - (np.mean(np.log(p / g))) ** 2) * 100.0

    return {
        "AbsRel": abs_rel, "RMSE": rmse, "Log10": log10,
        "d1": d1, "d2": d2, "d3": d3, "SILog": silog
    }


def scale_invariant_loss_torch(pred, target, mask=None):
    eps = 1e-6
    if mask is None:
        mask = target > 0
    mask = mask.float()
    valid = mask.sum(dim=[1,2,3])
    valid = torch.clamp(valid, min=1.0)

    d = torch.log(pred + eps) - torch.log(target + eps)
    d = d * mask
    mean_d2 = (d ** 2).sum(dim=[1,2,3]) / valid
    mean_d = d.sum(dim=[1,2,3]) / valid
    silog = torch.sqrt(torch.clamp(mean_d2 - 0.85 * (mean_d ** 2), min=0.0))
    return silog.mean()

class VKITTIDataset(Dataset):
    def __init__(self, rgb_paths, rgb_root, depth_root, img_size=(384,384), transforms_rgb=None):
        self.rgb_paths = rgb_paths
        self.rgb_root = str(Path(rgb_root).resolve())
        self.depth_root = str(Path(depth_root).resolve())
        self.img_size = tuple(img_size)
        self.transforms_rgb = transforms_rgb

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        rel = os.path.relpath(rgb_path, self.rgb_root)
        depth_rel = rel.replace(os.sep + "rgb" + os.sep, os.sep + "depth" + os.sep)
        depth_rel = depth_rel.replace("rgb_", "depth_")
        depth_path = os.path.join(self.depth_root, depth_rel)

        # ✅ Force .png for depth maps
        if depth_path.endswith(".jpg") or depth_path.endswith(".jpeg"):
            depth_path = depth_path.rsplit(".", 1)[0] + ".png"

        # Load RGB
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read RGB: {rgb_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA)

        # Load Depth (16-bit)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Cannot read depth: {depth_path}")
        depth = depth.astype(np.float32) / 100.0  # cm → m
        depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # To tensors
        if self.transforms_rgb:
            img_t = self.transforms_rgb(img)
        else:
            img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2,0,1)
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        return img_t, depth_t, rgb_path, depth_path

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
    def __init__(self, pretrained_path=None):
        super().__init__()
        backbone = resnet50(weights=None)
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            except TypeError:
                ckpt = torch.load(pretrained_path, map_location='cpu')
            backbone.load_state_dict(ckpt, strict=False)
            print("[INFO] Loaded pretrained weights (non-strict).")
        modules = list(backbone.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.decoder = DepthDecoder(2048)

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return F.softplus(out)

def find_rgb_files(rgb_root):
    return sorted(glob(os.path.join(rgb_root, "**", "rgb_*.jpg"), recursive=True))

def split_files(files, seed=42):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_test, n_val = int(0.2 * n), int(0.2 * n)
    n_train = n - n_val - n_test
    return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]

def collate_batch(batch):
    rgbs = torch.stack([b[0] for b in batch])
    depths = torch.stack([b[1] for b in batch])
    paths = [(b[2], b[3]) for b in batch]
    return rgbs, depths, paths

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_rgb = find_rgb_files(args.rgb_root)
    if len(all_rgb) == 0:
        raise RuntimeError(f"No RGB files found under {args.rgb_root}")

    train_files, val_files, test_files = split_files(all_rgb, args.seed)
    print(f"Dataset split → Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = VKITTIDataset(train_files, args.rgb_root, args.depth_root, (args.img_h, args.img_w), transform)
    val_ds   = VKITTIDataset(val_files, args.rgb_root, args.depth_root, (args.img_h, args.img_w), transform)
    test_ds  = VKITTIDataset(test_files, args.rgb_root, args.depth_root, (args.img_h, args.img_w), transform)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_batch)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_batch)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_batch)

    model = DepthModel(pretrained_path=args.pretrained).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-6)

    os.makedirs(args.out_dir, exist_ok=True)
    log_csv = os.path.join(args.out_dir, "train_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_RMSE","val_AbsRel","val_d1","val_d2","val_d3","val_SILog"])

    best_rmse = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for rgbs, depths, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            rgbs, depths = rgbs.to(device), depths.to(device)
            pred = model(rgbs)
            mask = (depths > 0).float()
            l1 = (torch.abs(pred - depths) * mask).sum() / mask.sum().clamp(min=1)
            silog = scale_invariant_loss_torch(pred, depths, mask)
            loss = l1 + 0.5 * silog
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        model.eval(); val_metrics = []
        with torch.no_grad():
            for rgbs, depths, _ in tqdm(val_dl, desc="Validating"):
                rgbs, depths = rgbs.to(device), depths.to(device)
                pred = model(rgbs)
                for i in range(pred.size(0)):
                    m = compute_metrics_numpy(pred[i,0].cpu().numpy(), depths[i,0].cpu().numpy())
                    if m: val_metrics.append(m)

        val_RMSE = np.mean([m["RMSE"] for m in val_metrics])
        val_AbsRel = np.mean([m["AbsRel"] for m in val_metrics])
        val_d1 = np.mean([m["d1"] for m in val_metrics])
        val_d2 = np.mean([m["d2"] for m in val_metrics])
        val_d3 = np.mean([m["d3"] for m in val_metrics])
        val_SILog = np.mean([m["SILog"] for m in val_metrics])

        print(f"Epoch {epoch}: TrainLoss={total_loss/len(train_dl):.4f}, Val RMSE={val_RMSE:.4f}")
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch,total_loss/len(train_dl),val_RMSE,val_AbsRel,val_d1,val_d2,val_d3,val_SILog])

        if val_RMSE < best_rmse:
            best_rmse = val_RMSE
            torch.save(model.state_dict(), os.path.join(args.out_dir,"best_model.pth"))
            print(f"[INFO] Best model saved (RMSE={best_rmse:.4f})")

    # Final test evaluation
    print("Testing best model...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir,"best_model.pth"), map_location=device))
    model.eval()
    test_metrics = []
    with torch.no_grad():
        for rgbs, depths, _ in tqdm(test_dl, desc="Testing"):
            rgbs, depths = rgbs.to(device), depths.to(device)
            pred = model(rgbs)
            for i in range(pred.size(0)):
                m = compute_metrics_numpy(pred[i,0].cpu().numpy(), depths[i,0].cpu().numpy())
                if m: test_metrics.append(m)

    for k in ["AbsRel","RMSE","Log10","d1","d2","d3","SILog"]:
        print(f"{k}: {np.mean([m[k] for m in test_metrics]):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_root", type=str, required=True)
    parser.add_argument("--depth_root", type=str, required=True)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results_dark")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=384)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
