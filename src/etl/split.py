from collections import defaultdict

import loguru

from .annotation import CONDITION_CLASSES, CONDITIONS_RU2EN, Example


def resolve_test_split(
    examples: list[Example],
    test_candidate_hashes: set[str],
    test_hashes: set[str] | None = None,
    min_n_objects: int | None = None,
    min_n_cases: int | None = None,
    verbose: bool = True,
) -> set[str]:
    out = set()
    n_out_objects_per_condition = defaultdict(int)

    if test_hashes is not None:
        for ex in examples:
            if ex["image_hash"] not in test_hashes:
                continue
            out.add(ex["image_hash"])
            for condition_ru, condition in CONDITIONS_RU2EN.items():
                if condition not in CONDITION_CLASSES:
                    continue
                objects = [obj for obj in ex["objects"] if obj["label"] == condition_ru]
                n_out_objects_per_condition[condition] += len(objects)

    hash_to_n_objects_per_condition = defaultdict(lambda: defaultdict(int))
    n_objects_per_condition = defaultdict(int)

    for ex in examples:
        if ex["image_hash"] in out or ex["image_hash"] not in test_candidate_hashes:
            continue
        for condition_ru, condition in CONDITIONS_RU2EN.items():
            if condition not in CONDITION_CLASSES:
                continue
            objects = [obj for obj in ex["objects"] if obj["label"] == condition_ru]
            n_objects_per_condition[condition] += len(objects)
            hash_to_n_objects_per_condition[ex["image_hash"]][condition] += len(objects)

    if min_n_objects is not None:
        for condition in sorted(
            n_objects_per_condition, key=lambda x: n_objects_per_condition[x]
        ):
            for (
                hash,
                n_objects_per_condition_case,
            ) in hash_to_n_objects_per_condition.items():
                if hash in out:
                    continue
                if n_out_objects_per_condition[condition] >= min_n_objects:
                    break
                if n_objects_per_condition_case[condition] > 0:
                    out.add(hash)
                    for cond, n_objects in n_objects_per_condition_case.items():
                        n_out_objects_per_condition[cond] += n_objects

    if min_n_cases is not None and len(out) < min_n_cases:
        for (
            hash,
            n_objects_per_condition_case,
        ) in hash_to_n_objects_per_condition.items():
            if hash in out:
                continue
            out.add(hash)
            for condition, n_objects in n_objects_per_condition_case.items():
                n_out_objects_per_condition[condition] += n_objects
            if len(out) == min_n_cases:
                break

    if verbose:
        loguru.logger.info(f"Total number of cases: {len(out)}")
        for condition in CONDITION_CLASSES:
            loguru.logger.info(f"{condition}: {n_out_objects_per_condition[condition]}")

    return out


def resolve_polygon_hashes(
    examples: list[Example],
    ignore_conditions: tuple[str, ...] = (
        "foreign_body",
        "periodontal_bone_loss",
        "prepared_tooth",
    ),
) -> list[str]:
    polygon_hashes = []
    for ex in examples:
        is_polygon = True
        for condition_ru, condition in CONDITIONS_RU2EN.items():
            if condition not in CONDITION_CLASSES or condition in ignore_conditions:
                continue
            objects = [obj for obj in ex["objects"] if obj["label"] == condition_ru]

            is_polygon = all(object_["shape"] == "polygon" for object_ in objects)
            if not is_polygon:
                break

        if is_polygon:
            polygon_hashes.append(ex["image_hash"])
    return polygon_hashes
