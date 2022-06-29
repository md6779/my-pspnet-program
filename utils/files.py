from __future__ import annotations

from pathlib import Path


def check_existence(fp: str | Path) -> None:
    """Checking file existance.

    Args:
        fp (str | Path): Path of file

    Raises:
        FileNotFoundError: File not found.
    """
    if isinstance(fp, str):
        fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"'{fp}' does not exist.")


def to_path(fp: str | Path) -> Path:
    """Convert to instance of `pathlib.Path`.

    Args:
        fp (str | Path): Path of file.

    Returns:
        Path: An instance of `pathlib.Path`.
    """
    return Path(fp) if isinstance(fp, str) else fp
