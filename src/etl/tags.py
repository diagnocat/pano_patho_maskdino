from .annotation import (
    CROWN_DESTRUCTION_CLASS_TO_LABEL,
    INVOLVEMENT_CLASS_TO_LABEL,
    IS_BUILDUP_CLASS_TO_LABEL,
    PBL_SEVERITY_CLASS_TO_LABEL,
    PBL_TYPE_CLASS_TO_LABEL,
    POST_MATERIAL_CLASS_TO_LABEL,
    SURFACES,
    Tag,
)


def resolve_tags(
    condition: str, tags: list[Tag], ignore_label: int = -100
) -> dict[str, int]:
    out = {}

    out["is_buildup"] = resolve_is_buildup(condition, tags, ignore_label)
    out["post_material"] = resolve_post_material(condition, tags, ignore_label)
    out["crown_destruction"] = resolve_crown_destruction(condition, tags, ignore_label)
    out["involvement"] = resolve_involvement_tag(condition, tags, ignore_label)
    out["pbl_severity"] = resolve_pbl_severity(condition, tags, ignore_label)
    out["pbl_type"] = resolve_pbl_type(condition, tags, ignore_label)

    surface_tags = resolve_surface_tags(condition, tags, ignore_label)
    out.update(surface_tags)

    return out


def resolve_is_buildup(
    condition: str, tags: list[Tag], ignore_label: int = -100
) -> int:
    if condition != "filling":
        return ignore_label
    is_buildup = resolve_tag_value_by_name(tags, "Билдап")
    if is_buildup is True:
        out = IS_BUILDUP_CLASS_TO_LABEL["Buildup"]
    elif is_buildup is False:
        out = IS_BUILDUP_CLASS_TO_LABEL["Not buildup"]
    elif is_buildup is None:
        out = ignore_label
    else:
        raise ValueError(f"Unknown is_buildup: {is_buildup}")
    return out


def resolve_post_material(
    condition: str, tags: list[Tag], ignore_label: int = -100
) -> int:
    out = ignore_label
    if condition == "post":
        if resolve_tag_value_by_name(tags, "Металлический штифт") is True:
            out = POST_MATERIAL_CLASS_TO_LABEL["Metal"]
        elif resolve_tag_value_by_name(tags, "Стекловолоконный штифт") is True:
            out = POST_MATERIAL_CLASS_TO_LABEL["Fiber"]
    return out


def resolve_surface_tags(
    condition: str, tags: list[Tag], ignore_label: int = -100
) -> dict[str, int]:
    tag_value_to_class = {
        "O": "occlusial",
        "M": "mesial",
        "D": "distal",
        "B": "buccal",
        "L": "lingual",
        "F": "vestibular",
        "I": "incisal",
        "Поверхность не определяется": "not_defined",
    }

    out = {f"is_surface_{surface}": ignore_label for surface in SURFACES}
    if condition in ("caries", "secondary_caries", "filling"):
        for tag_name, tag_class in tag_value_to_class.items():
            if (value := resolve_tag_value_by_name(tags, tag_name)) is not None:
                out[f"is_surface_{tag_class}"] = int(value)

    return out


def resolve_involvement_tag(
    condition: str, tags: list[Tag], ignore_label: int = -100
) -> int:
    tag_value_to_class = {
        "Не определяется": "not_defined",
        "Дентин и пульпа": "dentin_and_pulp",
        "Дентин": "dentin",
        "Эмаль": "enamel",
        "Корень": "root",
        "Инициальный": "initial",
    }

    out = ignore_label
    if condition in ("caries", "filling", "secondary_caries"):
        if (tag_value := resolve_tag_value_by_name(tags, "Вовлечение")) is not None:
            tag_class = tag_value_to_class.get(tag_value)
            if tag_class is not None:
                out = INVOLVEMENT_CLASS_TO_LABEL.get(tag_class, ignore_label)

    return out


def resolve_crown_destruction(
    condition: str, tags: list[Tag], ignore_label: int = -100
) -> int:
    tag_value_to_class = {
        "Менее 50%": "<50%",
        "Более 50%": ">50%",
    }
    out = ignore_label
    if condition in ("caries", "filling", "secondary_caries"):
        if (
            tag_value := resolve_tag_value_by_name(tags, "Объем разрушения")
        ) is not None:
            tag_class = tag_value_to_class.get(tag_value)
            if tag_class is not None:
                out = CROWN_DESTRUCTION_CLASS_TO_LABEL.get(tag_class, ignore_label)

    return out


def resolve_pbl_severity(
    condition: str, tags: list[Tag], ignore_label: int = -100
) -> int:

    out = ignore_label
    if condition != "periodontal_bone_loss":
        return out

    tag_value_to_class = {
        "Легкая": "mild",
        "Средняя": "moderate",
        "Тяжелая": "severe",
    }

    if (
        tag_value := resolve_tag_value_by_name(tags, "Степень пародонтита")
    ) is not None:
        tag_class = tag_value_to_class.get(tag_value)
        if tag_class is not None:
            out = PBL_SEVERITY_CLASS_TO_LABEL.get(tag_class, ignore_label)

    return out


def resolve_pbl_type(condition: str, tags: list[Tag], ignore_label: int = -100) -> int:

    out = ignore_label
    if condition != "periodontal_bone_loss":
        return out

    tag_value_to_class = {
        "Горизонтальный": "horizontal",
        "Вертикальный": "vertical",
        "Смешанный": "mixed",
    }

    if (tag_value := resolve_tag_value_by_name(tags, "Тип пародонтита")) is not None:
        tag_class = tag_value_to_class.get(tag_value)
        if tag_class is not None:
            out = PBL_TYPE_CLASS_TO_LABEL.get(tag_class, ignore_label)

    return out


def resolve_tag_value_by_name(tags: list[Tag], name: str) -> str | bool | None:
    for tag in tags:
        if tag["name"] == name:
            return tag["value"]

    return None
