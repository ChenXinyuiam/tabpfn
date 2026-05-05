"""Offline TabPFN setup helpers."""

import logging
import os
import shutil
from pathlib import Path


def setup_offline_tabpfn(local_model_dir: Path) -> None:
    """Patch Hugging Face downloads so TabPFN loads local model files."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(local_model_dir.parent)
    logging.getLogger("posthog").setLevel(logging.ERROR)

    required_files = {
        "tabpfn-v2.5-regressor-v2.5_default.ckpt": local_model_dir / "tabpfn-v2.5-regressor-v2.5_default.ckpt",
        "config.json": local_model_dir / "config.json",
    }
    for path in required_files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing local model file: {path}")

    def _patched_hf_hub_download(*args, **kwargs):
        repo_id = kwargs.get("repo_id", args[0] if args else None)
        filename = kwargs.get("filename", args[1] if len(args) > 1 else None)

        if repo_id != "Prior-Labs/tabpfn_2_5":
            raise FileNotFoundError(f"Offline patch: unsupported repo_id={repo_id}")
        if filename not in required_files:
            raise FileNotFoundError(f"Offline patch: no local mapping for {filename}")

        target_dir = Path(os.environ.get("APPDATA", local_model_dir.parent)) / "tabpfn"
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / f"__incoming__{filename}"
        shutil.copy2(required_files[filename], dst)
        print(f"[offline] {filename} -> {dst}")
        return str(dst)

    import huggingface_hub
    import huggingface_hub.file_download
    import tabpfn.model_loading as model_loading

    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    huggingface_hub.file_download.hf_hub_download = _patched_hf_hub_download
    model_loading.hf_hub_download = _patched_hf_hub_download
