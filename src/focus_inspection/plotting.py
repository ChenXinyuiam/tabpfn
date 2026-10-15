"""Shared plotting style for scientific focus figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def configure_fonts() -> None:
    for font_path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
