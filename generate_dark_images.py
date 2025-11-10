import os
import argparse
import random
import math
from glob import glob
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

DEFAULT_PARAMS = {
    "sF_min": 0.05,
    "sF_max": 0.8,
    "F_min": 0.5,
    "F_max": 3.5,
    "sb_min": 0.4,
    "sb_max": 1.0,
    "gF_min": 1.8,
    "gF_max": 2.2,
    "max_light_sources": 6,
    "max_flare_scale_px": 0.8,
    "gain_log_min": math.log(1.5),
    "gain_log_max": math.log(10.0),
    "read_noise_sigma": 6.0,
    "row_noise_sigma": 6.0,
    "quant_level": 256,
    "max_depth_clip_m": 655.35,
    "seed": 42,
}

def list_flare_images(flare_root):
    all_files = []
    for current_dir, subdirs, files in os.walk(flare_root):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(current_dir, file_name))
    return sorted(all_files)

def load_image_cv2(path, color=True):
    if color:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def save_image_rgb_uint8(path, img_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def copy_depth(src_depth_path, dst_depth_path):
    os.makedirs(os.path.dirname(dst_depth_path), exist_ok=True)
    import shutil
    shutil.copy2(src_depth_path, dst_depth_path)

def sample_log_uniform(a, b):
    return math.exp(random.uniform(math.log(a), math.log(b)))

def place_and_blend_flare(base_img, flare_img, position, scale, alpha=0.8, blur_radius=2):
    h, w = base_img.shape[:2]
    min_dim = min(h, w)
    target_size = max(1, int(min_dim * scale))
    
    fh, fw = flare_img.shape[:2]
    aspect = fw / fh
    if fw >= fh:
        new_w = target_size
        new_h = max(1, int(new_w / aspect))
    else:
        new_h = target_size
        new_w = max(1, int(new_h * aspect))
    
    flare_resized = cv2.resize(flare_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if flare_resized.shape[2] == 3:
        gray = cv2.cvtColor(flare_resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        alpha_mask = gray
    else:
        alpha_mask = flare_resized[..., 3].astype(np.float32) / 255.0
        flare_resized = flare_resized[..., :3]
    
    if blur_radius > 0:
        pil_img = Image.fromarray((alpha_mask * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        alpha_mask = np.array(pil_img).astype(np.float32) / 255.0

    cx, cy = int(position[0]), int(position[1])
    x0 = cx - new_w // 2
    y0 = cy - new_h // 2
    
    out = base_img.astype(np.float32).copy()
    overlay = flare_resized.astype(np.float32)
    am = np.clip(alpha_mask[..., None] * alpha, 0.0, 1.0)
    
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(w, x0 + new_w)
    y2 = min(h, y0 + new_h)
    
    if x1 >= x2 or y1 >= y2:
        return out.astype(np.uint8)
    
    fx1 = x1 - x0
    fy1 = y1 - y0
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)
    
    roi = out[y1:y2, x1:x2]
    ov = overlay[fy1:fy2, fx1:fx2]
    am_region = am[fy1:fy2, fx1:fx2]
    
    blended = (1 - am_region) * roi + am_region * np.clip(roi + ov, 0, 255)
    out[y1:y2, x1:x2] = blended
    
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_flare_module(img_rgb, flare_paths, params):
    h, w = img_rgb.shape[:2]
    base = img_rgb.astype(np.float32)
    
    sb = random.uniform(params["sb_min"], params["sb_max"])
    gF = random.uniform(params["gF_min"], params["gF_max"])
    base_dark = np.clip((sb * base / 255.0) ** (1.0 / gF) * 255.0, 0, 255).astype(np.uint8)
    
    F = sample_log_uniform(params["F_min"], params["F_max"])
    sF = sample_log_uniform(params["sF_min"], params["sF_max"])
    Nf = max(int(round(F / sF)), 1)
    Nf = min(Nf, params["max_light_sources"])
    
    if len(flare_paths) == 0:
        return base_dark
    
    out = base_dark.copy()
    
    for i in range(Nf):
        flare_path = random.choice(flare_paths)
        try:
            flare_img = load_image_cv2(flare_path, color=True)
        except:
            continue
        
        sFi = sample_log_uniform(params["sF_min"], params["sF_max"])
        
        margin_x = int(0.05 * w)
        margin_y = int(0.05 * h)
        cx = random.randint(margin_x, w - margin_x - 1)
        cy = random.randint(margin_y, h - margin_y - 1)
        
        scale_frac = min(params["max_flare_scale_px"], sFi)
        
        out = place_and_blend_flare(out, flare_img, (cx, cy), scale_frac, alpha=0.6, blur_radius=random.uniform(0.5, 3.0))
    
    out = np.clip((out / 255.0) ** gF * 255.0, 0, 255).astype(np.uint8)
    return out

def add_noise_physical(img_rgb, params):
    imf = img_rgb.astype(np.float32) / 255.0
    h, w = imf.shape[:2]
    
    gain = math.exp(random.uniform(params["gain_log_min"], params["gain_log_max"]))
    photon_scale = 30.0
    lam = np.clip(imf * gain * photon_scale, 0.0, None)
    
    lam_flat = lam.reshape(-1, 3)
    noisy_photons = np.random.poisson(lam_flat).astype(np.float32).reshape(h, w, 3)
    shot = noisy_photons / (gain * photon_scale)
    
    read_sigma = params["read_noise_sigma"] / 255.0
    read_noise = np.random.normal(0, read_sigma, size=(h, w, 3)).astype(np.float32)
    
    row_sigma = params["row_noise_sigma"] / 255.0
    row_offsets = np.random.normal(0, row_sigma, size=(h, 1, 1)).astype(np.float32)
    
    im_noisy = shot + read_noise + row_offsets
    im_noisy = np.clip(im_noisy, 0.0, 1.0)
    
    if params.get("quant_level", 256) < 256:
        levels = params["quant_level"]
        im_noisy = (np.round(im_noisy * (levels - 1)) / (levels - 1))
    
    im8 = (im_noisy * 255.0).astype(np.uint8)
    return im8

def process_single_frame(rgb_path, depth_path, vkitti_root, out_root, flare_paths, params):
    try:
        img = load_image_cv2(rgb_path, color=True)
    except Exception as e:
        print(f"Failed to load RGB image {rgb_path}: {e}")
        return False
    
    flared = apply_flare_module(img, flare_paths, params)
    noisy = add_noise_physical(flared, params)
    
    rel_path = os.path.relpath(rgb_path, vkitti_root)
    dst_rgb = os.path.join(out_root, rel_path)
    os.makedirs(os.path.dirname(dst_rgb), exist_ok=True)
    save_image_rgb_uint8(dst_rgb, noisy)
    
    if depth_path and os.path.exists(depth_path):
        rel_depth = os.path.relpath(depth_path, vkitti_root)
        dst_depth = os.path.join(out_root, rel_depth)
        copy_depth(depth_path, dst_depth)
    
    return True

def collect_vkitti_pairs(vkitti_root):
    rgb_files = []
    for root, dirs, files in os.walk(vkitti_root):
        for f in files:
            if f.startswith("rgb_") and f.lower().endswith(('.jpg', '.jpeg', '.png')):
                rgb_files.append(os.path.join(root, f))
    rgb_files = sorted(rgb_files)
    
    pairs = []
    for rgb_path in rgb_files:
        depth_candidate = rgb_path.replace(os.sep + "rgb" + os.sep, os.sep + "depth" + os.sep)
        depth_candidate = depth_candidate.replace("rgb_", "depth_")
        if os.path.exists(depth_candidate):
            pairs.append((rgb_path, depth_candidate))
        else:
            pairs.append((rgb_path, None))
    return pairs

def gen_task_wrapper(args):
    return process_single_frame(*args)

def generate_dataset(vkitti_root, flare_root, out_root, num_workers=4, params=None):
    random.seed(params.get("seed", 42))
    np.random.seed(params.get("seed", 42))
    
    flare_paths = list_flare_images(flare_root)
    if len(flare_paths) == 0:
        raise RuntimeError(f"No flare images found in {flare_root}. Did you point to the right folder?")
    
    pairs = collect_vkitti_pairs(vkitti_root)
    print(f"Found {len(pairs)} RGB images (depth available for {sum(1 for _, d in pairs if d)})")
    
    tasks = [(rgb, depth, vkitti_root, out_root, flare_paths, params) for rgb, depth in pairs]
    
    if num_workers <= 1:
        for task in tqdm(tasks):
            gen_task_wrapper(task)
    else:
        with Pool(processes=num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(gen_task_wrapper, tasks), total=len(tasks)):
                pass

def parse_args():
    parser = argparse.ArgumentParser(description="Create low-light version of Virtual KITTI using flares and realistic noise.")
    parser.add_argument('--vkitti_root', required=True, help='Root folder of Virtual KITTI 2 (contains SceneXX folders)')
    parser.add_argument('--flare_root', required=True, help='Folder containing Flare7K and Flare-R datasets')
    parser.add_argument('--out_root', required=True, help='Where to save the new low-light dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='How many CPU cores to use')
    parser.add_argument('--seed', type=int, default=DEFAULT_PARAMS["seed"], help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    params = DEFAULT_PARAMS.copy()
    params["seed"] = args.seed
    
    vkitti_root = args.vkitti_root
    flare_root = args.flare_root
    out_root = args.out_root
    num_workers = args.num_workers
    
    if not os.path.exists(vkitti_root):
        raise RuntimeError(f"VKITTI folder not found: {vkitti_root}")
    if not os.path.exists(flare_root):
        raise RuntimeError(f"Flare folder not found: {flare_root}")
    
    print("Looking for flare textures...")
    flare_paths = list_flare_images(flare_root)
    print(f"Found {len(flare_paths)} flare images")
    
    print("Starting low-light dataset generation...")
    generate_dataset(vkitti_root, flare_root, out_root, num_workers, params)
    print("All done! Your low-light VKITTI is ready.")

if __name__ == "__main__":
    main()
