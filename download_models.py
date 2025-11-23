#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tự động download models từ OpenVINO Model Zoo
"""

import os
import sys
import subprocess
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODELS = [
    {
        "name": "face-detection-retail-0004",
        "precision": "FP16",
        "files": [
            "face-detection-retail-0004.xml",
            "face-detection-retail-0004.bin"
        ]
    },
    {
        "name": "landmarks-regression-retail-0009",
        "precision": "FP16",
        "files": [
            "landmarks-regression-retail-0009.xml",
            "landmarks-regression-retail-0009.bin"
        ]
    },
    {
        "name": "face-reidentification-retail-0095",
        "precision": "FP16",
        "files": [
            "face-reidentification-retail-0095.xml",
            "face-reidentification-retail-0095.bin"
        ]
    }
]


def check_omz_tools():
    """Kiểm tra xem omz_downloader có sẵn không"""
    try:
        result = subprocess.run(
            ["omz_downloader", "--help"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_model(model_name, precision="FP16"):
    """Download model từ OpenVINO Model Zoo"""
    print(f"\n{'='*70}")
    print(f"Downloading {model_name} ({precision})...")
    print(f"{'='*70}")

    cmd = [
        "omz_downloader",
        "--name", model_name,
        "--output_dir", str(MODELS_DIR.parent / "intel"),
        "--precision", precision
    ]

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print(f"[OK] Downloaded {model_name}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {model_name}: {e}")
        return False


def copy_models_to_module():
    """Copy models vào thư mục models/"""
    print(f"\n{'='*70}")
    print("Copying models to module directory...")
    print(f"{'='*70}")

    intel_dir = MODELS_DIR.parent / "intel"

    for model_info in MODELS:
        model_name = model_info["name"]
        precision = model_info["precision"]
        model_dir = intel_dir / model_name / precision

        if not model_dir.exists():
            print(f"[WARNING] Model directory not found: {model_dir}")
            continue

        for file_name in model_info["files"]:
            src_file = model_dir / file_name
            dst_file = MODELS_DIR / file_name

            if src_file.exists():
                import shutil
                shutil.copy2(src_file, dst_file)
                print(f"[OK] Copied {file_name}")
            else:
                print(f"[WARNING] File not found: {src_file}")


def check_models_exist():
    """Kiểm tra xem models đã có chưa"""
    all_exist = True
    missing = []

    print(f"\n{'='*70}")
    print("Checking existing models...")
    print(f"{'='*70}")

    for model_info in MODELS:
        for file_name in model_info["files"]:
            file_path = MODELS_DIR / file_name
            if file_path.exists():
                print(f"[OK] {file_name}")
            else:
                print(f"[MISSING] {file_name}")
                all_exist = False
                missing.append(file_name)

    return all_exist, missing


def main():
    """Main function"""
    print("="*70)
    print("FACE RECOGNITION MODULE - MODEL DOWNLOADER".center(70))
    print("="*70)

    # Check existing models
    all_exist, missing = check_models_exist()

    if all_exist:
        print(f"\n[OK] All models are already downloaded!")
        print("You can run: python facere.py")
        return 0

    print(f"\n[INFO] Missing {len(missing)} model file(s)")

    # Check omz_downloader
    if not check_omz_tools():
        print("\n" + "="*70)
        print("[ERROR] OpenVINO Model Zoo tools not found!")
        print("="*70)
        print("\nPlease install openvino-dev:")
        print("  pip install openvino-dev")
        print("\nOr download models manually from:")
        print("  https://github.com/openvinotoolkit/open_model_zoo")
        return 1

    # Download models
    print(f"\n{'='*70}")
    print("Starting download...")
    print(f"{'='*70}")

    for model_info in MODELS:
        download_model(model_info["name"], model_info["precision"])

    # Copy to module
    copy_models_to_module()

    # Final check
    print("\n")
    all_exist, missing = check_models_exist()

    if all_exist:
        print(f"\n{'='*70}")
        print("[SUCCESS] All models downloaded successfully!".center(70))
        print(f"{'='*70}")
        print("\nYou can now run:")
        print("  python facere.py")
        return 0
    else:
        print(f"\n{'='*70}")
        print(f"[WARNING] Still missing {len(missing)} file(s)".center(70))
        print(f"{'='*70}")
        print("\nPlease check the error messages above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
