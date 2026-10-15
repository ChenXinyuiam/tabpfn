"""Isolated, status-tracked output directories; completed runs are never overwritten."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .provenance import file_sha256, write_json


def latest_artifact(root: Path, filename: str) -> Path:
    """Resolve only a completed run, with a legacy flat-directory fallback."""
    import json

    pointer = root / "latest.json"
    if pointer.exists():
        run = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"]).resolve()
        if not run.is_relative_to((root / "runs").resolve()):
            raise ValueError("Latest run pointer escapes its output root")
        status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError("Latest artifact must belong to a completed run")
        path = run / filename
    else:
        path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}; run feature extraction or provide an explicit path")
    return path


def snapshot_code(out: Path, extra_sources=()):
    """Preserve the exact local source, including uncommitted changes."""
    import shutil

    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src/focus_inspection").glob("*.py")),
        root / "utils/provenance.py",
        root / "utils/runs.py",
        root / "pyproject.toml",
        root / "uv.lock",
        *[Path(path).resolve() for path in extra_sources],
    ]
    manifest = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        manifest[relative.as_posix()] = file_sha256(target)
    write_json(out / "source_snapshot/manifest.json", manifest)


@contextmanager
def experiment_run(root: Path, kind: str, *, resume: bool = False, extra_sources=()):
    root = Path(root)
    if resume:
        import json

        status_path = root / "run_status.json"
        if not status_path.is_file():
            raise ValueError("Resume requires the exact interrupted run directory")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["kind"] != kind or status["status"] == "complete":
            raise ValueError("Only an incomplete run of the same workflow can be resumed")
        out = root
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = root / "runs" / f"{stamp}-{uuid4().hex[:8]}"
        out.mkdir(parents=True, exist_ok=False)
        snapshot_code(out, extra_sources=extra_sources)
    status = {"kind": kind, "status": "running", "output_dir": str(out.resolve())}
    write_json(out / "run_status.json", status)
    print(f"Run directory: {out.resolve()}", flush=True)
    try:
        yield out
    except BaseException as error:
        write_json(out / "run_status.json", {**status, "status": "failed", "error": str(error)})
        raise
    else:
        write_json(out / "run_status.json", {**status, "status": "complete"})
        write_json(out.parent.parent / "latest.json", {"run_dir": str(out.resolve()), "kind": kind})
