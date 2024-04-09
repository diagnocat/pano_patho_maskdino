import os
import shutil
from pathlib import Path
from typing import Collection, Dict, List, Union

import loguru
import typer
from typing_extensions import TypeAlias

from src.defs import ROOT

SOURCE_DPATH = ROOT.joinpath("src")
DL_PATH = SOURCE_DPATH.joinpath("dl")
ETL_DPATH = SOURCE_DPATH.joinpath("etl")

SyncBranch: TypeAlias = Union[str, "SyncTree", Collection[Union[str, "SyncTree"]]]
SyncTree: TypeAlias = Dict[os.PathLike, "SyncBranch"]

SYNC_TREE: SyncTree = {
    DL_PATH: [
        {"inference": ["driver.py", "postprocess.py", "nms.py"]},
    ],
    ETL_DPATH: ["annotation.py"],
}  # type: ignore


def find_sync_paths(sync_tree: "SyncTree") -> List[Path]:
    """ """
    out = []
    for root, branch in sync_tree.items():
        out.extend(_find_sync_paths_recursively_impl(root, branch))
    return out


def _find_sync_paths_recursively_impl(
    root: os.PathLike, branch: "SyncBranch"
) -> List[Path]:
    """ """
    assert (root := Path(root)).is_absolute()

    out = []
    if isinstance(branch, dict):
        for branch_name, sub_branch in branch.items():
            assert (new_root := root.joinpath(branch_name)).is_dir(), new_root
            out.extend(_find_sync_paths_recursively_impl(new_root, sub_branch))

    elif isinstance(branch, str):
        path = root.joinpath(branch)
        assert path.is_file() or path.is_dir(), path
        out.append(path)

    elif isinstance(branch, Collection):
        for el in branch:
            out.extend(_find_sync_paths_recursively_impl(root, el))

    else:
        raise TypeError(branch)

    return out


def sync_content(
    target_dpath: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Directory where to put the code",
    ),
) -> None:
    """ """
    if not target_dpath.exists():
        target_dpath.mkdir()

    for sync_subroot in map(Path, SYNC_TREE):
        try:
            assert sync_subroot.is_dir()
            shutil.rmtree(target_dpath.joinpath(sync_subroot.name))

        except:
            ...

    for source_path in find_sync_paths(SYNC_TREE):
        suffix = source_path.relative_to(SOURCE_DPATH)
        target_path = target_dpath.joinpath(suffix)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        loguru.logger.info(
            f"Transferred source from {str(source_path):80} to {str(target_path):80}"
        )

        if source_path.is_dir():
            shutil.copytree(source_path, target_path)

        elif source_path.is_file():
            shutil.copyfile(source_path, target_path)

        else:
            raise RuntimeError(source_path)


if __name__ == "__main__":
    typer.run(sync_content)
