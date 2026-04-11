"""TractCloud model data download and cache management."""

import logging
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GITHUB_RELEASE = (
    "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0"
)
_MODEL_ARCHIVE = "TrainedModel.tar.gz"
_TRAIN_DATA_ARCHIVE = "TrainData_800clu800ol.tar.gz"


def data_dir():
    """Return the directory where TractCloud model data is cached.

    Uses TRACTCLOUD_DATA_DIR env var if set, otherwise ~/.cache/tractcloud/.
    """
    d = os.environ.get(
        "TRACTCLOUD_DATA_DIR",
        os.path.join(Path.home(), ".cache", "tractcloud"),
    )
    os.makedirs(d, exist_ok=True)
    return d


def _download_file(url, dest_path, progress_callback=None):
    """Download a file with optional progress reporting."""
    def _reporthook(blocknum, blocksize, totalsize):
        if progress_callback and totalsize > 0:
            fraction = min(blocknum * blocksize / totalsize, 1.0)
            progress_callback(fraction)
    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)


def ensure_model_data(progress_callback=None):
    """Download and extract model weights and atlas data if not present.

    Returns:
        (weight_path, args_path, mass_center) tuple.
    """
    dd = data_dir()

    # Trained model
    model_dir = os.path.join(dd, "TrainedModel")
    weight_path = os.path.join(model_dir, "best_tract_f1_model.pth")
    args_path = os.path.join(model_dir, "cli_args.txt")

    if not os.path.exists(weight_path):
        logger.info("Downloading TractCloud trained model...")
        archive = os.path.join(dd, _MODEL_ARCHIVE)
        _download_file(
            f"{_GITHUB_RELEASE}/{_MODEL_ARCHIVE}", archive,
            progress_callback)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=dd)
        os.remove(archive)

    # HCP mass center
    mass_center_path = os.path.join(
        dd, "TrainData_800clu800ol", "HCP_mass_center.npy")
    if not os.path.exists(mass_center_path):
        logger.info("Downloading HCP atlas data (for mass center)...")
        archive = os.path.join(dd, _TRAIN_DATA_ARCHIVE)
        _download_file(
            f"{_GITHUB_RELEASE}/{_TRAIN_DATA_ARCHIVE}", archive,
            progress_callback)
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if "HCP_mass_center.npy" in m.name]
            if members:
                tar.extractall(path=dd, members=members)
            else:
                tar.extractall(path=dd)
        os.remove(archive)

    mass_center = np.load(mass_center_path)
    return weight_path, args_path, mass_center
