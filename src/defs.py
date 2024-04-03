from pathlib import Path

ROOT = Path(__file__).absolute().parent.parent

DATA_PATH = ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"

PROCESSED_DATA_PATH = DATA_PATH / "processed"
PROCESSED_IMAGES_PATH = PROCESSED_DATA_PATH / "images"

CROPS_PATH = DATA_PATH / "crops.json"

HASHES_PATH = DATA_PATH / "hashes"
INDEX_HASHES_PATH = HASHES_PATH / "index_hashes.json"
