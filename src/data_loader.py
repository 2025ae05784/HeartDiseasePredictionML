import os
import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_loader")

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
RAW_PATH = os.path.join("data", "raw", "processed.cleveland.data")

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

def fetch_uci_dataset(url: str = UCI_URL, destination: str = RAW_PATH) -> pd.DataFrame:
    """Downloads raw Cleveland heart disease data from UCI repository if not cached locally."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        logger.info(f"Fetching dataset from {url}...")
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        with open(destination, "wb") as f:
            f.write(res.content)
        logger.info(f"Dataset successfully stored at {destination}")
    else:
        logger.info(f"Using locally cached raw dataset from {destination}")

    df = pd.read_csv(destination, header=None, names=FEATURE_COLUMNS, na_values="?")
    logger.info(f"Loaded raw dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    fetch_uci_dataset()