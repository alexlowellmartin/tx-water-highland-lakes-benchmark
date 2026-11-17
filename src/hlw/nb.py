"""Shared setup for the teaching notebooks in notebooks/."""

from __future__ import annotations

import warnings

# set at import so they also cover imports made before setup() is called
warnings.filterwarnings("ignore", module="tqdm")
warnings.filterwarnings("ignore", message=".*IProgress.*")

BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b"


def setup():
    """Inline figures (some hlw modules select the Agg backend on import), a quiet, consistent
    plot style, and no noise from optional-dependency warnings."""
    warnings.filterwarnings("ignore", module="tqdm")
    warnings.filterwarnings("ignore", message=".*IProgress.*")
    import matplotlib.pyplot as plt

    # these modules select the Agg backend on import: import them first, then override
    import hlw.analyze
    import hlw.blogfigs  # noqa: F401

    try:
        get_ipython().run_line_magic("matplotlib", "inline")
    except NameError:  # plain Python
        pass
    plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9.5,
                         "axes.titlesize": 10, "axes.titlelocation": "left"})
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
