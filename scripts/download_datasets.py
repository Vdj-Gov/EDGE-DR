"""
EDGE-DR — Dataset Download Helper
===================================
Downloads APTOS 2019, EyePACS, and IDRiD using the Kaggle API.
Automatically configures authentication from .env or .kaggle/access_token.

Usage:
    python scripts/download_datasets.py --dataset aptos
    python scripts/download_datasets.py --dataset idrid
    python scripts/download_datasets.py --dataset eyepacs
    python scripts/download_datasets.py --dataset all
"""

import os
import sys
import zipfile
import click
from pathlib import Path

# Set up Kaggle configuration paths before importing KaggleApi
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = PROJECT_ROOT / ".kaggle"
TOKEN_FILE = KAGGLE_DIR / "access_token"

os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_DIR))

if TOKEN_FILE.exists() and "KAGGLE_API_TOKEN" not in os.environ:
    os.environ["KAGGLE_API_TOKEN"] = TOKEN_FILE.read_text().strip()

from kaggle.api.kaggle_api_extended import KaggleApi

DATASETS = {
    "aptos": {
        "type": "competition",
        "name": "aptos2019-blindness-detection",
        "out_dir": "data/aptos2019",
    },
    "idrid": {
        "type": "dataset",
        "name": "mariaherrerot/idrid-dataset",
        "out_dir": "data/idrid",
    },
    "eyepacs": {
        "type": "competition",
        "name": "diabetic-retinopathy-detection",
        "out_dir": "data/eyepacs",
    },
}


def extract_all_zips(directory: Path):
    """Extract any downloaded .zip files and remove them."""
    for zf_path in directory.glob("*.zip"):
        print(f"Extracting {zf_path.name} to {directory}...")
        try:
            with zipfile.ZipFile(zf_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            zf_path.unlink()
            print(f"Extracted and removed {zf_path.name}")
        except Exception as e:
            print(f"Warning: could not extract {zf_path.name}: {e}")


def download_dataset_item(api: KaggleApi, ds_key: str):
    info = DATASETS[ds_key]
    out_dir = PROJECT_ROOT / info["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"Downloading {ds_key.upper()} ({info['name']})")
    print(f"Destination: {out_dir}")
    print(f"==========================================")

    if info["type"] == "competition":
        api.competition_download_files(
            competition=info["name"],
            path=str(out_dir),
            force=False,
            quiet=False
        )
    else:
        api.dataset_download_files(
            dataset=info["name"],
            path=str(out_dir),
            unzip=True,
            force=False,
            quiet=False
        )

    extract_all_zips(out_dir)
    print(f"✓ {ds_key.upper()} download and extraction completed.")


@click.command()
@click.option(
    "--dataset",
    default="aptos",
    type=click.Choice(["aptos", "idrid", "eyepacs", "all"]),
    help="Dataset to download. 'aptos' (approx 9 GB) is recommended to start."
)
def main(dataset):
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    try:
        api.authenticate()
        user = api.get_config_value("username")
        print(f"Authenticated successfully as: {user}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    targets = ["aptos", "idrid"] if dataset == "all" else [dataset]
    for target in targets:
        try:
            download_dataset_item(api, target)
        except Exception as e:
            print(f"Error downloading {target}: {e}")

    print("\nDownload process finished.")
    print("Next step: Run the preprocessing pipeline:")
    print("  python -m src.data.preprocess --config configs/config.yaml")


if __name__ == "__main__":
    main()
