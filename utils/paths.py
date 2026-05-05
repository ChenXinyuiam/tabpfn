"""Path helpers used by pipeline scripts."""

from pathlib import Path
from collections.abc import Iterable


def project_root() -> Path:
    """返回上上级的目录

    parents[0] → /home/user/project/app
    parents[1] → /home/user/project
    parents[2] → /home/user
    parents[3] → /home
    parents[4] → /
    Returns:
        Path: _description_
    """
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_excel_files(directory: Path, suffix: str = ".xlsx") -> list[Path]:
    return sorted(
        path
        for path in directory.glob(f"*{suffix}")
        if path.is_file() and not path.name.startswith("~$")
    )


def iter_excel_files_recursive(directory: Path, suffix: str = ".xlsx") -> list[Path]:
    return sorted(
        path
        for path in directory.rglob(f"*{suffix}")
        if path.is_file() and not path.name.startswith("~$")
    )


def batch_dirs(root: Path) -> list[Path]:
    """Return child batch directories when present, otherwise the root itself."""
    children = sorted(path for path in root.iterdir() if path.is_dir())
    return children or [root]


def select_batch_dir(root: Path, batch: str | None = None) -> Path:
    """Select one batch directory. Defaults to the latest child directory."""
    if batch:
        selected = root / batch
        if selected.is_dir():
            return selected
        matches = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(batch))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise FileExistsError(f"Batch prefix {batch!r} is ambiguous under {root}: {names}")
        raise FileNotFoundError(f"Batch directory not found: {selected}")

    children = sorted(path for path in root.iterdir() if path.is_dir())
    return children[-1] if children else root


def resolve_existing_path(path: Path, candidates: Iterable[Path]) -> Path:
    if path.exists():
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path
