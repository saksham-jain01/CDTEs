"""Evaluate two 401(k) witness profiles using one shared fitted estimator."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save the figure without requiring a graphical session.
import matplotlib.pyplot as plt
import numpy as np

from skcd.witness import (
    compute_all_witness_functions,
    load_pension_data,
    plot_witness_functions,
)


PROFILES = [28, 53]             # Indices in the held-out evaluation set
BOOTSTRAP_DRAWS = 1000


def main():
    root = Path(__file__).resolve().parents[1]
    data = load_pension_data(root / "data" / "pension.csv")
    x, a, y = data["X"], data["A"], data["Y"]
    half = len(x) // 2
    result = compute_all_witness_functions(
        x[:half], a[:half], y[:half], x[half:], a[half:], y[half:],
        data["eval_X"], PROFILES, np.linspace(-3, 3, 100), data["outcome_cols"],
        alpha=0.05, B=BOOTSTRAP_DRAWS,
    )
    result["alpha"] = 0.05
    figure, _ = plot_witness_functions(result, data["frame"], data["outcome_cols"])
    output = root / "witness_401k.png"
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")
    return result


if __name__ == "__main__":
    main()
