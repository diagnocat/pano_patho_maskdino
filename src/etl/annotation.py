from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np


class Example(TypedDict):
    annotated_classes: list[str]
    last_task_id: int
    last_task_name: str
    image_path: str
    image_hash: str
    image_height: int
    image_width: int
    objects: list[Object]


class Object(TypedDict):
    author: Author
    last_modified: str
    shape: Literal["bbox", "polygon", "line", "point", "skeleton"]
    points: list[list[int]] | dict[PointLabel, Point]
    uid: str
    label: str
    tags: list[Tag]
    origin: str


PointLabel: TypeAlias = Literal[
    "Bone distal", "CEJ mesial", "CEJ distal", "Bone mesial"
]


class Point(TypedDict):
    points: list[list[int]]
    tags: list[Tag]


class Tag(TypedDict):
    author: Author | None
    last_modified: str | None
    name: str
    value: str | bool | None


class Author(TypedDict):
    name: str
    uid: str


class ItemMeta(TypedDict):
    image_path: Path
    image_hash: str
    mask_path: Path


class Item(TypedDict):
    numeration: np.ndarray
    tooth_type: np.ndarray
    tag: np.ndarray


# Primary classification task
CONDITION_CLASSES = (
    "artificial_crown",
    "canal_filling",
    "caries",
    "cast_post_and_core",
    "dental_calculus",
    "external_resorption",
    "filling",
    "foreign_body",
    "furcation_lesion",
    "implant",
    "lack_of_interproximal_contact",
    "missed_canal",
    "open_margin",
    "orthodontic_appliances",
    "overfilling",
    "overhang",
    "periapical_lesion",
    "periodontal_bone_loss",
    "post",
    "pontic",
    "prepared_tooth",
    "pulp_stone",
    "secondary_caries",
    "short_filling",
    "voids_in_filling",
    "voids_present_in_the_root_filling",
)

CONDITION_CLASS_TO_LABEL = {
    c: label for label, c in enumerate(CONDITION_CLASSES, start=1)
}
CONDITION_LABEL_TO_CLASS = {label: c for c, label in CONDITION_CLASS_TO_LABEL.items()}
# `CONDITION_CLASS_TO_LABEL` mapping is for COCO dataset, where categories
# start from 1. However, predictions start from 0.
# It's handy to have this mapping as well.
CONDITION_CLASS_TO_LABEL_PREDICTIONS = {
    c: label for label, c in enumerate(CONDITION_CLASSES)
}
CONDITION_LABEL_TO_CLASS_PREDICTIONS = {
    label: c for c, label in CONDITION_CLASS_TO_LABEL_PREDICTIONS.items()
}


# Secondary classification tasks (aka tags)

IS_BUILDUP_CLASSES = ("Buildup", "Not buildup")
IS_BUILDUP_CLASS_TO_LABEL = {c: label for label, c in enumerate(IS_BUILDUP_CLASSES)}
IS_BUILDUP_LABEL_TO_CLASS = {label: c for c, label in IS_BUILDUP_CLASS_TO_LABEL.items()}

POST_MATERIAL_CLASSES = ("Fiber", "Metal")
POST_MATERIAL_CLASS_TO_LABEL = {
    c: label for label, c in enumerate(POST_MATERIAL_CLASSES)
}
POST_MATERIAL_LABEL_TO_CLASS = {
    label: c for c, label in POST_MATERIAL_CLASS_TO_LABEL.items()
}

SURFACES = (
    "distal",
    "occlusial",
    "lingual",
    "mesial",
    "vestibular",
    "incisal",
    "buccal",
    "not_defined",
)
SURFACES_CLASSES = {surface: (f"Not {surface}", surface) for surface in SURFACES}
SURFACES_CLASS_TO_LABEL = {
    surface: {c: label for label, c in enumerate(classes)}
    for surface, classes in SURFACES_CLASSES.items()
}
SURFACES_LABEL_TO_CLASS = {
    surface: {label: c for label, c in enumerate(classes)}
    for surface, classes in SURFACES_CLASSES.items()
}

INVOLVEMENT_CLASSES = (
    "dentin",
    "dentin_and_pulp",
    "initial",
    "not_defined",
    "enamel",
    "root",
)
INVOLVEMENT_CLASS_TO_LABEL = {c: label for label, c in enumerate(INVOLVEMENT_CLASSES)}
INVOLVEMENT_LABEL_TO_CLASS = {
    label: c for c, label in INVOLVEMENT_CLASS_TO_LABEL.items()
}

CROWN_DESTRUCTION_CLASSES = (
    "<50%",
    ">50%",
)
CROWN_DESTRUCTION_CLASS_TO_LABEL = {
    c: label for label, c in enumerate(CROWN_DESTRUCTION_CLASSES)
}
CROWN_DESTRUCTION_LABEL_TO_CLASS = {
    label: c for c, label in CROWN_DESTRUCTION_CLASS_TO_LABEL.items()
}

PBL_SEVERITY_CLASSES = (
    "mild",
    "moderate",
    "severe",
)
PBL_SEVERITY_CLASS_TO_LABEL = {c: label for label, c in enumerate(PBL_SEVERITY_CLASSES)}
PBL_SEVERITY_LABEL_TO_CLASS = {
    label: c for c, label in PBL_SEVERITY_CLASS_TO_LABEL.items()
}

PBL_TYPE_CLASSES = (
    "horizontal",
    "vertical",
    "mixed",
)
PBL_TYPE_CLASS_TO_LABEL = {c: label for label, c in enumerate(PBL_TYPE_CLASSES)}
PBL_TYPE_LABEL_TO_CLASS = {label: c for c, label in PBL_TYPE_CLASS_TO_LABEL.items()}


TAG_TO_LABEL_TO_CLASS = {
    "is_buildup": IS_BUILDUP_LABEL_TO_CLASS,
    "post_material": POST_MATERIAL_LABEL_TO_CLASS,
    "involvement": INVOLVEMENT_LABEL_TO_CLASS,
    "crown_destruction": CROWN_DESTRUCTION_LABEL_TO_CLASS,
    **{
        f"is_surface_{surface}": label_to_class
        for surface, label_to_class in SURFACES_LABEL_TO_CLASS.items()
    },
}

CONDITIONS_RU2EN = {
    "Культя": "prepared_tooth",
    "Периимплантит": "peri-implantitis",
    "Режущий край": "incisal_part_of_the_tooth",
    "Коронковая часть зуба": "crown_part_of_the_tooth",
    "Ортодонтическая конструкция": "orthodontic_appliances",
    "Эндодонтический доступ": "endodontic_access",
    "Искусственная коронка": "artificial_crown",
    "Накладка на зуб": "artificial_crown",
    "Пломбировочный материал в канале": "canal_filling",
    "Кариес / Разрушение коронки": "caries",
    "Вкладка культевая": "cast_post_and_core",
    "Кламп": "clamp",
    "Зубной камень": "dental_calculus",
    "Эмаль зуба": "enamel",
    "Эндодонтический инструмент": "endodontic_instrument",
    "Наружная резорбция корня": "external_resorption",
    "Пломба": "filling",
    "Инородное тело": "foreign_body",
    "Поражение фуркации": "furcation_lesion",
    "Формирователь десны": "healing_abutment",
    "Горизонтальный перелом корня": "horizontal_root_fracture",
    "Гиперцементоз": "hypercementosis",
    "Имплантат": "implant",
    "Внутренняя резорбция корня": "internal_resorption",
    "Отсутствующий контактный пункт": "lack_of_interproximal_contact",
    "Нижнечелюстной канал": "mandibular_canal",
    "Пустой канал": "missed_canal",
    "Зуб отсутствует": "missing_tooth",
    "Зуб": "tooth",
    "Облитерация канала": "obliteration",
    "Вторичный кариес": "secondary_caries",
    "Нарушение прилегания": "open_margin",
    "Пора в пломбе": "voids_in_filling",
    "Шина / Ретейнер": "orthodontic_appliance",
    "Перепломбировка канала": "overfilling",
    "Нависающий край": "overhang",
    "Периодонтит": "periapical_lesion",
    "Уплотнение костной ткани": "osteosclerosis",
    "Пародонтит": "periodontal_bone_loss",
    "Признаки пародонтита": "periodontal_bone_loss",
    "Промежуточная часть мостовидного протеза": "pontic",
    "Штифт": "post",
    "Пульпарная камера": "pulp_chamber",
    "Пульпотомия": "pulpotomy",
    "Дентикль": "pulp_stone",
    "Корень зуба": "root",
    "Канал зуба": "root_canal",
    "Перфорация": "root_perforation",
    "Недопломбировка канала": "short_filling",
    "Вертикальный перелом корня": "vertical_root_fracture",
    "Недостаточная плотность пломбировочного материала": "voids_present_in_the_root_filling",
    "Нарушение прилегания штифтовой конструкции": "voids_present_in_the_root_filling",
    "Абатмент": "abatment",
}
