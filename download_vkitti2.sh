#!/usr/bin/env bash
set -euo pipefail

# ===============================
# download_vkitti2_strict.sh
# Robust downloader for Virtual KITTI 2 (v2.0.3) RGB + Depth
# ===============================

# Target directory (change if you want)
OUT_DIR="${PWD}vkitti2"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

# NAVER Labs download base (observed working path)
BASE_URL="http://download.europe.naverlabs.com//virtual_kitti_2.0.3"

# Filenames we want
FILES=("vkitti_2.0.3_rgb.tar" "vkitti_2.0.3_depth.tar")

# Download tool selection:
# - If aria2c available, use it (faster / parallel / resume-friendly).
# - Otherwise fallback to wget with sensible options.
if command -v aria2c >/dev/null 2>&1; then
  echo "[*] Using aria2c for downloads (recommended)"
  ARIA_OPTS=(
    "--max-concurrent-downloads=2"
    "--split=4"
    "--min-split-size=10M"
    "--continue=true"
    "--retry-wait=5"
    "--max-tries=10"
    "--retry-on-http-4xx=true"
    "--check-integrity=true"
  )
  for f in "${FILES[@]}"; do
    url="${BASE_URL}/${f}"
    echo "[*] Downloading $f from $url"
    aria2c "${ARIA_OPTS[@]}" -d "$OUT_DIR" -o "$f" "$url" || {
      echo "[!] aria2c failed for $f — falling back to wget for this file"
      wget --continue --tries=10 --timeout=30 --tries=10 --retry-connrefused \
        --waitretry=5 --no-clobber --user-agent="Mozilla/5.0 (X11; Linux x86_64)" \
        -O "$f" "$url"
    }
  done
else
  echo "[*] aria2c not found — using wget"
  WGET_OPTS=(--continue --tries=12 --timeout=30 --waitretry=5 --retry-connrefused \
             --user-agent="Mozilla/5.0 (X11; Linux x86_64)" --no-check-certificate)
  for f in "${FILES[@]}"; do
    url="${BASE_URL}/${f}"
    echo "[*] Downloading $f from $url"
    wget "${WGET_OPTS[@]}" -O "$f" "$url"
  done
fi

# Verify files exist and are non-empty
echo "[*] Verifying downloaded files..."
for f in "${FILES[@]}"; do
  if [ ! -s "$f" ]; then
    echo "[ERROR] File $f is missing or empty in $(pwd). Download likely failed."
    echo "        Please check network, VPN, or try again later."
    exit 2
  else
    echo "[OK] $f downloaded (size: $(stat -c%s "$f") bytes)."
  fi
done

# Extract archives safely
echo "[*] Extracting archives (tar -xvf)..."
for f in "${FILES[@]}"; do
  echo "  -> Extracting $f ..."
  tar -xvf "$f"
done

# Optional: remove tar files (uncomment if you want to free space)
CLEAN_UP=false
if [ "$CLEAN_UP" = true ] ; then
  echo "[*] Cleaning up tar files..."
  for f in "${FILES[@]}"; do
    rm -f "$f"
  done
fi

echo "✅ Done. Virtual KITTI2 should be at: $OUT_DIR"
