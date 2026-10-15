"""Read-only checks for maintained project workflows."""

import importlib
import json
from importlib.metadata import version

from src.cli import COMMANDS
from src.focus_inspection.settings import DEFAULT_MODEL_PATH, DEFAULT_RAW_DIR


def inspect_environment():
    checks = []
    modules = (
        "src.data.template_schema",
        "src.data.stages",
        "src.focus_inspection.dataset",
        "src.focus_inspection.optimize_focus",
        "src.focus_inspection.ablate_focus",
        "src.focus_inspection.calibrate_focus",
        "src.focus_inspection.predict_focus",
        "src.regression.fd_to_fl",
        "src.regression.evaluation",
        "src.data.audit",
        "src.geometry.triangle",
        "src.physics.force",
        "src.evaluation.system",
        "src.workflows.system",
    )
    command_modules = (module for module, _ in COMMANDS.values() if module != __name__)
    for name in dict.fromkeys((*modules, *command_modules)):
        try:
            importlib.import_module(name)
            checks.append({"check": name, "ok": True})
        except Exception as error:
            checks.append({"check": name, "ok": False, "error": str(error)})
    for name in ("tabpfn", "torch", "numpy", "pandas", "scipy", "scikit-learn", "openpyxl"):
        try:
            checks.append({"check": name, "ok": True, "version": version(name)})
        except Exception as error:
            checks.append({"check": name, "ok": False, "error": str(error)})
    checks.append(
        {"check": "classifier_checkpoint", "ok": DEFAULT_MODEL_PATH.is_file(), "path": str(DEFAULT_MODEL_PATH)}
    )
    checks.append({"check": "default_focus_data", "ok": DEFAULT_RAW_DIR.is_dir(), "path": str(DEFAULT_RAW_DIR)})
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    return {"checks": checks, "cuda_available": cuda_available, "ok": all(row["ok"] for row in checks)}


def main():
    result = inspect_environment()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
