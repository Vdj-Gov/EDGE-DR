"""
EDGE-DR — APTOS Auto Extractor & Preprocessor
==============================================
Monitors the downloading aptos2019-blindness-detection.zip file,
waits until download completes, extracts the train_images, and
triggers dataset preprocessing.
"""

import time
import zipfile
import subprocess
import sys
from pathlib import Path

APTOS_DIR = Path("data/aptos2019")
ZIP_FILE = APTOS_DIR / "aptos2019-blindness-detection.zip"
TARGET_SIZE = 10215289875  # ~9.51 GB


def wait_for_download():
    print(f"Monitoring {ZIP_FILE.name}...")
    last_size = -1
    stable_count = 0

    while True:
        if not ZIP_FILE.exists():
            print("Waiting for download to begin...")
            time.sleep(5)
            continue

        curr_size = ZIP_FILE.stat().st_size
        pct = (curr_size / TARGET_SIZE) * 100
        print(f"Progress: {curr_size / (1024**3):.2f} GB / {TARGET_SIZE / (1024**3):.2f} GB ({pct:.1f}%)")

        if curr_size == last_size and curr_size >= TARGET_SIZE * 0.99:
            stable_count += 1
            if stable_count >= 3:
                # File size has been completely stable for 3 checks and is near target
                print("Download appears complete! Verifying zip integrity...")
                try:
                    with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
                        first_bad = zf.testzip()
                        if first_bad is None:
                            print("Zip integrity verified successfully!")
                            return True
                except Exception as e:
                    print(f"Zip not yet ready ({e}), waiting...")
                    stable_count = 0
        else:
            stable_count = 0

        last_size = curr_size
        time.sleep(10)


def extract_aptos():
    print("Extracting train_images from zip...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
        # Extract train.csv and train_images
        namelist = zf.namelist()
        train_members = [m for m in namelist if m.startswith("train_images/") or m == "train.csv"]
        print(f"Extracting {len(train_members)} training items...")
        zf.extractall(APTOS_DIR, members=train_members)
    print("Extraction complete!")


def main():
    if not (APTOS_DIR / "train_images").exists() or len(list((APTOS_DIR / "train_images").glob("*.png"))) == 0:
        wait_for_download()
        extract_aptos()

    print("\nStarting APTOS preprocessing...")
    cmd = [sys.executable, "-m", "src.data.preprocess", "--config", "configs/config.yaml", "--dataset", "all"]
    subprocess.run(cmd, check=True)
    print("\nPreprocessing complete! dataset.csv is ready for training.")


if __name__ == "__main__":
    main()

