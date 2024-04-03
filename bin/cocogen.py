import argparse

from src.defs import PROCESSED_DATA_PATH
from src.etl.cocogen import prepare_coco_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="coco")
    args = parser.parse_args()
    prepare_coco_dataset(PROCESSED_DATA_PATH / args.dataset_name)
