"""Unified command entry point with lazy workflow imports."""

import argparse
import runpy
import sys

COMMANDS = {
    "system": ("src.workflows.system", "Audit, grouped evaluation, model bundles and new-triangle inference"),
    "doctor": ("src.doctor", "Check imports, dependencies and local model availability"),
    "data": ("src.data.pipeline", "Run or inspect the data preparation pipeline"),
    "focus-plots": ("src.focus_inspection.plot_fzong_sign_by_re", "Plot signs for Excel x R x Re samples"),
    "focus-audit": (
        "src.focus_inspection.audit_focus",
        "Audit input geometry, points and labels without model fitting",
    ),
    "focus-features": ("src.focus_inspection.build_focus_boundary_features", "Extract boundary and spatial features"),
    "focus-baseline": ("src.focus_inspection.train_focus_tabpfn", "Reproduce the original grouped baseline"),
    "focus-optimize": ("src.focus_inspection.optimize_focus", "Run nested grouped TabPFN selection"),
    "focus-ablate": (
        "src.focus_inspection.ablate_focus",
        "Compare width and midpoint with identical budgets and seeds",
    ),
    "focus-calibrate": (
        "src.focus_inspection.calibrate_focus",
        "Evaluate training-only grouped probability calibration",
    ),
    "focus-predict": ("src.focus_inspection.predict_focus", "Predict a new batch using a saved recipe"),
    "focus-review": ("src.focus_inspection.review_focus", "Review unique samples and geometry-level uncertainty"),
    "regression-evaluate": ("src.regression.evaluation", "Evaluate FD/FL spatial and geometry holdouts"),
    "fd-optimize": ("src.regression.optimize_fd", "Nested unseen-geometry FD-u optimization benchmark"),
    "fd-tabpfn-tune": ("src.regression.tune_tabpfn_fd", "TabPFN-only nested FD-u tuning"),
    "fd-rotation-audit": ("src.regression.rotation_dataset", "Audit all provider rotation exports"),
    "fd-rotation-tune": ("src.regression.tune_rotated_fd", "Nested TabPFN rotation-aware FD-u tuning"),
    "fd-rotation-report": ("src.regression.rotation_report", "Rebuild rotation benchmark figures and metrics"),
    "fd-transfer-tune": ("src.regression.transfer_fd", "Nested mesh-coordinate and target-scaling FD experiments"),
    "fd-transfer-ensemble": ("src.regression.transfer_ensemble", "Combine inner-selected FD recipes"),
    "fd-transfer-audit": ("src.regression.transfer_diagnostics", "Inspect suspicious FD conditions without editing labels"),
    "fd-transfer-report": ("src.regression.transfer_report", "Compare saved FD transfer experiments"),
    "fd-zero-audit": ("src.regression.zero_fd", "Audit zeros and training-only replacement sensitivity"),
    "regression-predict": ("src.regression.predict_fields", "Predict native FD-u and FL on paired positions with audited statistics"),
    "regression-report": ("src.regression.prediction_report", "Rebuild FD-u/FL statistics without model fitting"),
    "fd-to-fl": ("src.regression.fd_to_fl", "Predict FD u at matching FL positions"),
    "fl-at-fd": ("src.regression.fl_at_fd", "Predict FL at FD positions"),
    "fd-from-fl": ("src.regression.fd_from_fl", "Predict FD u using interpolated FL"),
    "interpolate": ("src.regression.interpolation", "Interpolate predicted files per Re"),
    "interpolate-raw-fl": ("src.regression.raw_fl_interpolation", "Interpolate raw FL/FLX per workbook and Re"),
    "interpolate-raw-fl-tabpfn": ("src.regression.raw_fl_tabpfn", "Predict raw FL/FLX interpolation points with TabPFN"),
    "interpolate-fly": ("src.regression.fly_tabpfn", "Predict FLy midpoints and four-corner centers with TabPFN"),
    "interpolate-fly-linear": ("src.regression.fly_midpoints", "Arithmetic FLy interpolation reference"),
    "fl-lines": ("src.regression.fl_lines", "Reproduce the two reference reverse-FL lines at FD positions"),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="FD/FL prediction and focus classification workflows")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.epilog = "\n".join(f"{name}: {description}" for name, (_, description) in COMMANDS.items())
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    original_argv = sys.argv
    try:
        sys.argv = [COMMANDS[command][0], *args[1:]]
        runpy.run_module(COMMANDS[command][0], run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
