import json
import random

import numpy as np

from src.defs import HASHES_PATH, INDEX_HASHES_PATH, RAW_DATA_PATH
from src.etl.annotation import CONDITION_CLASSES
from src.etl.split import resolve_polygon_hashes, resolve_test_split
from src.etl.utils import get_examples_lakefs, translate_annotated_classes


def main() -> None:

    np.random.seed(42)
    random.seed(42)

    with open(HASHES_PATH / "pipelines_test_hashes.txt", "r") as f:
        pipelines_test_hashes = set(f.read().splitlines())

    examples = get_examples_lakefs(RAW_DATA_PATH)

    annotated_examples = []
    for example in examples:
        annotated_classes = translate_annotated_classes(example["annotated_classes"])
        if any(c not in annotated_classes for c in CONDITION_CLASSES):
            continue
        annotated_examples.append(example)

    annotated_hashes = set(ex["image_hash"] for ex in annotated_examples)

    polygon_annotated_hashes = set(resolve_polygon_hashes(annotated_examples))

    test_hashes = resolve_test_split(
        examples=annotated_examples,
        test_candidate_hashes=polygon_annotated_hashes - pipelines_test_hashes,
        test_hashes=None,
        min_n_objects=50,
        min_n_cases=300,
        verbose=True,
    )
    print()

    val_hashes = resolve_test_split(
        examples=annotated_examples,
        test_candidate_hashes=polygon_annotated_hashes
        - test_hashes
        - pipelines_test_hashes,
        min_n_objects=30,
        verbose=False,
    )

    val_hashes = resolve_test_split(
        examples=annotated_examples,
        test_hashes=val_hashes,
        test_candidate_hashes=annotated_hashes
        - test_hashes
        - val_hashes
        - pipelines_test_hashes,
        min_n_objects=30,
        verbose=True,
        min_n_cases=150,
    )

    annotated_data_hashes = {
        "annotated": {
            "description": "Cases with all required conditions present in `annotated_classes`",
            "hashes": list(annotated_hashes),
        },
        "polygonal": {
            "description": "Annotated cases with polygons",
            "hashes": list(polygon_annotated_hashes),
        },
        "val": {
            "hashes": list(val_hashes),
            "description": "Val cases (polygonal & fully annotated).",
        },
        "test": {
            "description": "Test cases (polygonal & fully annotated)",
            "hashes": list(test_hashes),
        },
        "pipelines_test_cases": {
            "description": "Test set used in pipelines for pano pathology model",
            "hashes": list(pipelines_test_hashes),
        },
    }

    with open(INDEX_HASHES_PATH, "w") as f:
        json.dump(annotated_data_hashes, f)


if __name__ == "__main__":
    main()
