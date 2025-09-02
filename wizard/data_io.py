import os
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn import datasets

# -----------------
# Built-in samples
# -----------------
def load_sample(name: str) -> Tuple[pd.DataFrame, str, str]:
    """Return (df, target, task_type)."""
    if name == "Iris":
        data = datasets.load_iris(as_frame=True)
        return data.frame, data.target.name, "classification"
    if name == "Wine":
        data = datasets.load_wine(as_frame=True)
        return data.frame, data.target.name, "classification"
    if name == "Diabetes":
        data = datasets.load_diabetes(as_frame=True)
        return data.frame, "target", "regression"
    if name == "Wine Quality":
        from wizard.data_io import load_wine_quality
        return load_wine_quality()
    raise ValueError("Unknown sample")

def infer_types(df: pd.DataFrame):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    cats = [c for c in df.columns if c not in nums]
    return nums, cats

# -----------------
# Wine Quality (red+white from UCI)
# -----------------
DATA_DIR = "data"
RED_LOCAL = os.path.join(DATA_DIR, "winequality-red.csv")
WHITE_LOCAL = os.path.join(DATA_DIR, "winequality-white.csv")

RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _download_csv(url: str, dest_path: str, sep: str = ";") -> bool:
    try:
        df = pd.read_csv(url, sep=sep)
        df.to_csv(dest_path, index=False, sep=sep)  # <-- important
        return True
    except Exception as e:
        print(f"[wine] Download failed from {url}: {e}")
        return False

def load_wine_quality(cache_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, str, str]:
    """
    Load Wine Quality dataset (red + white) from local CSVs or UCI.
    Adds 'wine_type' (0=red, 1=white).
    Returns (df, target_column, task_type).
    """
    _ensure_dir(cache_dir)

    have_red = os.path.exists(RED_LOCAL)
    have_white = os.path.exists(WHITE_LOCAL)

    if not (have_red and have_white):
        if not have_red:
            print("[wine] red CSV missing locally; attempting download…")
            have_red = _download_csv(RED_URL, RED_LOCAL)
        if not have_white:
            print("[wine] white CSV missing locally; attempting download…")
            have_white = _download_csv(WHITE_URL, WHITE_LOCAL)

    if not (have_red and have_white):
        raise RuntimeError(
            "Wine CSVs not found and download failed.\n"
            "Fix: add these files under ./data/ :\n"
            "  - data/winequality-red.csv\n"
            "  - data/winequality-white.csv\n"
            "Get them from the UCI Wine Quality dataset."
        )

    red = pd.read_csv(RED_LOCAL, sep=";")
    white = pd.read_csv(WHITE_LOCAL, sep=";")
    red["wine_type"] = 0
    white["wine_type"] = 1
    df = pd.concat([red, white], ignore_index=True)

    target = "quality"
    task_type = "classification"  # default; user can flip to classification later
    return df, target, task_type
