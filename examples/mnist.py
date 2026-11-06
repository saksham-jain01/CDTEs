"""Run one test on the bundled MNIST embeddings; edit the settings below."""
from pathlib import Path

import numpy as np

from skcd import proposed_mmd_wald_test


N = 250
METHOD = "mmd"                 # "mmd" or "wald"
ALTERNATIVE = True             # False selects null outcomes
KNOWN_PROPENSITY = False       # False fits the propensity models
BOOTSTRAP_DRAWS = 1000


def main():
    directory = Path(__file__).resolve().parents[1] / "data" / "mnist"
    x = np.load(directory / "null_unedited.npy", allow_pickle=False)
    outcome = "alternate_bright_rand_rot_det.npy" if ALTERNATIVE else "null_bright_rand.npy"
    y = np.load(directory / outcome, allow_pickle=False)
    a = np.load(directory / "trt_assign.npy", allow_pickle=False)
    propensity = np.load(directory / "propensities.npy", allow_pickle=False)

    # One sampled dataset, with replacement as in the simulation design.
    rows = np.random.default_rng().choice(len(x), size=N, replace=True)
    result = proposed_mmd_wald_test(
        x[rows], a[rows], y[rows], test_type=METHOD,
        propensity=propensity[rows] if KNOWN_PROPENSITY else None,
        n_bootstrap=BOOTSTRAP_DRAWS, alpha=0.05,
    )
    statistic, p_value, rejected, _ = result
    print(f"{METHOD.upper()}: statistic={statistic:.6g}, p={p_value:.4g}, reject={rejected}")
    return result


if __name__ == "__main__":
    main()
