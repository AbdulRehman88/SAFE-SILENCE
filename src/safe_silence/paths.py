"""Repository-rooted paths independent of the terminal working directory."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def from_root(relative: str | Path) -> Path:
    return (project_root() / relative).resolve()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve an absolute path unchanged or a relative path from the project root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else from_root(path)
