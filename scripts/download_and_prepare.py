"""Script to download and verify MovieLens 1M dataset."""

import logging
import ssl
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "ml-1m.zip"
TARGET_DIR = RAW_DIR / "ml-1m"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (TARGET_DIR / "ratings.dat").exists() and (TARGET_DIR / "movies.dat").exists():
        logger.info("MovieLens 1M already exists at %s", TARGET_DIR)
        return

    logger.info("Downloading MovieLens 1M from %s...", URL)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx) as resp, open(ZIP_PATH, "wb") as f:
        while chunk := resp.read(128 * 1024):
            f.write(chunk)

    logger.info("Extracting %s...", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)

    logger.info("MovieLens 1M successfully prepared in %s", TARGET_DIR)


if __name__ == "__main__":
    main()
