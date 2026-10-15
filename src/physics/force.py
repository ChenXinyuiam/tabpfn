"""Signed component addition with dimensional checks and explicit provenance."""

import numpy as np

FORCE_TO_N = {"N": 1.0, "mN": 1e-3, "uN": 1e-6, "nN": 1e-9, "pN": 1e-12}
VELOCITY_TO_M_S = {"m/s": 1.0, "mm/s": 1e-3, "um/s": 1e-6}


class PhysicsUnavailable(ValueError):
    """An explicit physical prerequisite has not been provided."""


def validate_physics(targets, config):
    if not config or config.get("confirmed") is not True:
        raise PhysicsUnavailable("Confirmed Fzong formula and units have not been supplied")
    if config.get("formula") != "signed_component_sum" or not config.get("version") or not config.get("reference"):
        raise PhysicsUnavailable("Expected versioned signed_component_sum with a scientific reference")
    if config.get("output_unit") not in FORCE_TO_N or config.get("component") not in {"x", "z"}:
        raise PhysicsUnavailable("An explicit force output unit and x/z component are required")
    for task in ("FD", "FL"):
        target = targets.get(task, {})
        if target.get("component") != config["component"]:
            raise PhysicsUnavailable(task + " component does not match Fzong")
        if target.get("quantity") == "force" and target.get("unit") in FORCE_TO_N:
            continue
        conversion = config.get("fd_conversion", {})
        if task == "FD" and target.get("quantity") == "velocity" and target.get("unit") in VELOCITY_TO_M_S:
            gain = conversion.get("gain_N_per_m_s")
            if (
                conversion.get("confirmed") is True
                and conversion.get("reference")
                and conversion.get("kind") == "declared_linear"
                and isinstance(gain, (int, float))
                and np.isfinite(gain)
                and gain > 0
            ):
                continue
        raise PhysicsUnavailable(task + " is not a confirmed force (or explicitly converted velocity)")


def compose_force(fd, fl, targets, config):
    validate_physics(targets, config)
    fd, fl = np.asarray(fd, dtype=float), np.asarray(fl, dtype=float)
    if fd.shape != fl.shape or fd.ndim != 1 or not np.isfinite(fd).all() or not np.isfinite(fl).all():
        raise ValueError("Force composition requires finite aligned one-dimensional arrays")
    spec = targets["FD"]
    factor = (
        FORCE_TO_N[spec["unit"]]
        if spec["quantity"] == "force"
        else VELOCITY_TO_M_S[spec["unit"]] * config["fd_conversion"]["gain_N_per_m_s"]
    )
    result = (fd * factor + fl * FORCE_TO_N[targets["FL"]["unit"]]) / FORCE_TO_N[config["output_unit"]]
    if not np.isfinite(result).all():
        raise ValueError("Force conversion overflow")
    return result


def reference_force(values, target, config):
    if target.get("quantity") != "force" or target.get("component") != config["component"]:
        raise ValueError("Fzong reference is not the same signed force component")
    return np.asarray(values, dtype=float) * FORCE_TO_N[target["unit"]] / FORCE_TO_N[config["output_unit"]]
