# SKCD

Implementation and data for the paper **Conditional Distributional Treatment
Effects: Doubly Robust Estimation and Testing**, Saksham Jain and Alex Luedtke,
[ICML 2026 / arXiv](https://arxiv.org/abs/2603.16829v2).

## Install and run

From this checkout, with Python 3.9 or newer:

```bash
python -m pip install -e .
python examples/mnist.py
python examples/pension.py
```

The MNIST example runs one test on a sampled dataset. Edit its settings to choose
MMD or Wald, null or alternative outcomes, sample size, bootstrap count, and
estimated or known propensities. The 401(k) example fits one witness estimator,
evaluates two held-out profiles, and saves `witness_401k.png`.

The methods select CUDA when available and otherwise use CPU. Full 401(k)
calculations use dense matrices; the original implementation ran on an NVIDIA
T4 with 32 GB system RAM. On macOS, LightGBM may require `brew install libomp`.

## Use your own data

Supply covariates `X` with shape `(n, d)`, binary treatment `A` with shape `(n,)`,
and outcomes `Y` with shape `(n,)` or `(n, q)`:

```python
from skcd import proposed_mmd_test, proposed_wald_test

statistic, p_value, rejected, details = proposed_mmd_test(
    X, A, Y, n_bootstrap=1000, alpha=0.05
)
# Replace proposed_mmd_test with proposed_wald_test for the Wald statistic.
```

Rows must correspond across arrays. The methods use the first and second halves
as cross-fitting folds; independently sampled/shuffled rows are assumed.
Propensity models are fitted by default. Supply `propensity=probabilities` to use
known probabilities. Defaults are Gaussian kernels with median bandwidths and
KRR regularization at 0.001. Results include the bootstrap statistics and bandwidths.

For Nyström approximation, add `kernel_approximation="nystrom"`,
`nystrom_rank_x=10`, and `nystrom_rank_y=10` to either call. The package also
exports `proposed_mmd_bandwidth_sweep_test` and
`proposed_wald_bandwidth_sweep_test` for bandwidth aggregation.

The KCD baseline method (https://arxiv.org/abs/2102.08208) is included with its required utility functions:
```python
from skcd import kcd_baseline_test

statistic, p_value = kcd_baseline_test(X, A, Y, num_perm=150)
```

## Contents

| File | Role |
| --- | --- |
| `skcd/proposed_test.py` | MMD/Wald tests, bootstrap and bandwidth aggregation |
| `skcd/utils.py` | Kernels, nuisance estimation and exact/Nyström matrix calculations |
| `skcd/witness.py` | Shared witness estimation, confidence bands and 401(k) loading/plotting |
| `skcd/kcd_baseline.py` | KCD comparison method |
| `skcd/__init__.py` | Public imports |
| `examples/mnist.py` | One direct test using the included embeddings |
| `examples/pension.py` | Direct 401(k) witness example |
| `data/` | Five MNIST arrays and the 401(k) CSV, described below |

The data is ready for analysis; image encoder training is not required. 

## Included data

| File | Content |
| --- | --- |
| `data/mnist/null_unedited.npy` | Covariates: 45,000 × 5 float32 image embeddings |
| `data/mnist/null_bright_rand.npy` | Null outcomes: 45,000 × 5 float32 embeddings |
| `data/mnist/alternate_bright_rand_rot_det.npy` | Alternative outcomes: 45,000 × 5 float32 embeddings |
| `data/mnist/trt_assign.npy` | 45,000 int64 treatment assignments |
| `data/mnist/propensities.npy` | 45,000 float32 treatment probabilities |
| `data/pension.csv` | 9,915-household, 44-column SIPP 401(k) dataset |

The five MNIST files share the same row ordering and must be used together.
They contain derived embeddings and assignments; raw digit images, labels,
encoder weights, and PCA state are not needed to use these inputs. 

The direct example `examples/mnist.py` loads these arrays and samples one dataset
on which to run the method. Users can change the test or sample size in that
file, or pass their own arrays to the package functions.

The 401(k) data comes from Wave 4 of the 1990 Survey of Income and Program
Participation, as described in the paper. The code uses treatment `e401`;
outcomes `net_tfa`, `net_nifa`, `tw`; continuous covariates `age`, `inc`, `fsize`,
`educ`; and indicators `db`, `marr`, `twoearn`, `pira`, `hown`. The other CSV
columns are retained from the original input. The original 99/9,816 evaluation
and fitting split is made by the analysis code, and is held fixed.
