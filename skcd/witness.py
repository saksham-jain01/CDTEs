"""
Witness Function Analysis with Slice-Based Confidence Bands

Computes rigorous fixed-x confidence bands by bootstrapping the
influence function restricted to the RKHS slice H_x, while preserving
the exact geometry of the SKCD estimator.
"""


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import clip_propensity, median_bandwidth as simulation_median_bandwidth
from sklearn.model_selection import train_test_split
import optuna_integration.lightgbm as lgb
import optuna
import time
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BANDWIDTH_MAX_SAMPLES = 2000


# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

def rbf_kernel_gram(X1, X2=None, sigma=1.0):
    """RBF kernel Gram matrix."""
    if not isinstance(X1, torch.Tensor):
        X1 = torch.tensor(X1, device=device, dtype=torch.float32)
    if X2 is None:
        X2 = X1
    elif not isinstance(X2, torch.Tensor):
        X2 = torch.tensor(X2, device=device, dtype=torch.float32)

    if X1.ndim == 1: X1 = X1.unsqueeze(1)
    if X2.ndim == 1: X2 = X2.unsqueeze(1)

    x1_sq = X1.pow(2).sum(dim=-1, keepdim=True)
    x2_sq = X2.pow(2).sum(dim=-1, keepdim=True)
    sq_dists = x1_sq + x2_sq.T - 2 * (X1 @ X2.T)
    sq_dists = sq_dists.clamp_min_(0.0)

    return torch.exp(-sq_dists / (2 * sigma ** 2))


def median_bandwidth(data, max_samples=BANDWIDTH_MAX_SAMPLES):
    """Use the simulation median heuristic on at most 2,000 fitting rows.

    Sample without replacement before calling the shared implementation:
    sigma = sqrt(median(d^2)), with PyTorch's lower-middle median convention.
    Callers pool both fitting folds and exclude the held-out evaluation profiles.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if len(data) > max_samples:
        idx = np.random.choice(len(data), max_samples, replace=False)
        data = data[idx]
    if len(data) < 2:
        return 1.0
    return max(simulation_median_bandwidth(data, device=device), 1e-6)


def standardize_data(data):
    """Standardize to zero mean and unit variance."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-8] = 1.0
    return (data - mean) / std


def fit_propensity_model(X, A):
    """
    Fit propensity score P(A=1 | X) using LightGBM with Optuna tuning.
    """
    # Create internal validation set for early stopping/tuning
    train_x, val_x, train_y, val_y = train_test_split(X, A, test_size=0.2, random_state=42)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "force_row_wise": True # Suppress overhead warning
    }

    train_data = lgb.Dataset(train_x, label=train_y)
    val_data = lgb.Dataset(val_x, label=val_y)

    # Use LightGBM Tuner to find best hyperparameters automatically
    tuner = lgb.LightGBMTuner(
        params,
        train_data,
        valid_sets=[val_data],
        show_progress_bar=False,
        callbacks=[lgb.early_stopping(10, verbose=False)]
    )

    # Suppress tuning output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuner.run()

    model = tuner.get_best_booster()

    return model


# ==============================================================================
# 2. C MATRIX COMPUTATION (Double Robustness Structure)
# ==============================================================================

def compute_C_matrix(X1, A1, X2, A2, pi1, pi2, K, reg=0.001):
    """
    Compute the C matrix encoding the influence function structure.
    Data must be sorted: treated first, then control, within each fold.
    """
    n1, n2 = len(X1), len(X2)
    n = n1 + n2

    n1t, n1c = int(np.sum(A1)), int(n1 - np.sum(A1))
    n2t, n2c = int(np.sum(A2)), int(n2 - np.sum(A2))

    pi1 = clip_propensity(pi1)
    pi2 = clip_propensity(pi2)

    # IPW ratios
    ratio_1t = A1 / pi1
    ratio_1c = (1 - A1) / (1 - pi1)
    ratio_2t = A2 / pi2
    ratio_2c = (1 - A2) / (1 - pi2)

    # Diagonal blocks
    C11 = np.diag((1 / (2 * n1)) * (ratio_1t - ratio_1c))
    C22 = np.diag((1 / (2 * n2)) * (ratio_2t - ratio_2c))

    # Regression adjustment coefficients
    coef_1t = (1 / (2 * n1)) * (1 - ratio_1t)
    coef_1c = -(1 / (2 * n1)) * (1 - ratio_1c)
    coef_2t = (1 / (2 * n2)) * (1 - ratio_2t)
    coef_2c = -(1 / (2 * n2)) * (1 - ratio_2c)

    reg_eye = lambda m: reg * torch.eye(m, device=K.device, dtype=K.dtype)

    # Beta matrices (kernel regression weights)
    beta_2t = torch.linalg.solve(
        K[n1:n1+n2t, n1:n1+n2t] + reg_eye(n2t),
        K[:n1, n1:n1+n2t].T
    ).T.cpu().numpy()

    beta_2c = torch.linalg.solve(
        K[n1+n2t:n, n1+n2t:n] + reg_eye(n2c),
        K[:n1, n1+n2t:n].T
    ).T.cpu().numpy()

    beta_1t = torch.linalg.solve(
        K[:n1t, :n1t] + reg_eye(n1t),
        K[n1:, :n1t].T
    ).T.cpu().numpy()

    beta_1c = torch.linalg.solve(
        K[n1t:n1, n1t:n1] + reg_eye(n1c),
        K[n1:, n1t:n1].T
    ).T.cpu().numpy()

    # Off-diagonal blocks
    C12 = np.hstack((coef_1t[:, None] * beta_2t, coef_1c[:, None] * beta_2c))
    C21 = np.hstack((coef_2t[:, None] * beta_1t, coef_2c[:, None] * beta_1c))

    return np.block([[C11, C12], [C21, C22]])


# ==============================================================================
# 3. SHARED FITTING AND SLICE-BASED CONFIDENCE BANDS
# ==============================================================================

def prepare_witness_context(X_1, A_1, Y_1, X_2, A_2, Y_2,
                            y_grid, y_dims, sigma_x, sigma_y, reg=0.001):
    """Fit nuisances and prepare matrices once for all evaluation profiles.

    Each propensity model is trained on one fitting fold and evaluated on the
    other. C, C L C^T, and the outcome-grid kernels do not depend on a profile x.
    Evaluation profiles never enter this shared fitting step.
    """
    n_1, n_2 = len(X_1), len(X_2)

    # Sort data: treated first, then control (within each fold)
    sort_idx1 = np.argsort(A_1)[::-1]
    sort_idx2 = np.argsort(A_2)[::-1]

    X1_s, A1_s, Y1_s = X_1[sort_idx1], A_1[sort_idx1], Y_1[sort_idx1]
    X2_s, A2_s, Y2_s = X_2[sort_idx2], A_2[sort_idx2], Y_2[sort_idx2]

    X_sorted = np.vstack([X1_s, X2_s])
    Y_sorted = np.vstack([Y1_s, Y2_s])

    # Fit propensity models
    pi_1 = fit_propensity_model(X_2, A_2).predict(X1_s)
    pi_2 = fit_propensity_model(X_1, A_1).predict(X2_s)

    # Kernel Gram matrices
    X_t = torch.tensor(X_sorted, device=device, dtype=torch.float32)
    Y_t = torch.tensor(Y_sorted, device=device, dtype=torch.float32)

    K = rbf_kernel_gram(X_t, sigma=sigma_x)
    L = rbf_kernel_gram(Y_t, sigma=sigma_y)  # Full Y kernel

    # C matrix (encodes double robustness structure)
    C_np = compute_C_matrix(X1_s, A1_s, X2_s, A2_s, pi_1, pi_2, K, reg)
    C = torch.tensor(C_np, device=device, dtype=torch.float32)

    # Outcome covariance: M = C L C^T
    M = C @ L @ C.T

    # Outcome-grid kernels are also common to every evaluation profile.
    L_grids = {}
    for y_dim in y_dims:
        Y_grid_points = np.zeros((len(y_grid), Y_sorted.shape[1]))
        Y_grid_points[:, y_dim] = y_grid
        Y_grid_t = torch.tensor(Y_grid_points, device=device, dtype=torch.float32)
        L_grids[y_dim] = rbf_kernel_gram(Y_t, Y_grid_t, sigma=sigma_y)

    # K and L can be released after constructing C and M; profiles only need
    # these derived matrices and the much smaller outcome-grid kernels.
    return {
        'X_t': X_t,
        'C': C,
        'M': M,
        'L_grids': L_grids,
        'n_1': n_1,
        'n_2': n_2,
        'sort_idx1': sort_idx1,
        'sort_idx2': sort_idx2,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
    }


def compute_slice_band_and_witness(context, eval_x, alpha=0.05, B=1000):
    """Evaluate one profile using shared fitted matrices, without refitting.

    Each profile has its own slice Gram matrix, bootstrap critical value, and
    band width. That band is used for all its outcome cross-sections.
    """
    X_t, C, M = context['X_t'], context['C'], context['M']
    n_1, n_2 = context['n_1'], context['n_2']
    sort_idx1, sort_idx2 = context['sort_idx1'], context['sort_idx2']
    sigma_x, sigma_y = context['sigma_x'], context['sigma_y']
    n = n_1 + n_2
    profile_device = X_t.device
    eval_x = np.asarray(eval_x).reshape(1, -1)

    # Kernel vector at eval_x: k_vec[i] = k(X_i, x)
    eval_x_t = torch.tensor(eval_x, device=profile_device, dtype=torch.float32)
    k_vec = rbf_kernel_gram(X_t, eval_x_t, sigma=sigma_x).view(-1)

    # Slice Gram matrix: G_ij = k(X_i, x) k(X_j, x) M_ij
    G_slice = torch.outer(k_vec, k_vec) * M

    # =========================================================================
    # BOOTSTRAP: Compute q_hat (once, using full Y structure)
    # =========================================================================
    print(f"    Running {B} bootstrap iterations...")

    T_bootstrap = np.zeros(B)

    for b in range(B):
        # Multinomial multipliers (per fold)
        W_1 = np.random.multinomial(n_1, [1/n_1] * n_1)
        W_2 = np.random.multinomial(n_2, [1/n_2] * n_2)

        # Reorder to match sorted indices
        xi_1 = torch.tensor(W_1[sort_idx1] - 1, device=profile_device, dtype=torch.float32)
        xi_2 = torch.tensor(W_2[sort_idx2] - 1, device=profile_device, dtype=torch.float32)
        xi = torch.cat([xi_1, xi_2])

        # T^{(b)} = xi^T G_slice xi
        T_b = (xi @ G_slice @ xi).item()
        T_bootstrap[b] = T_b

    q_hat = np.percentile(T_bootstrap, 100 * (1 - alpha))
    band_width = np.sqrt(q_hat)

    print(f"    q_hat = {q_hat:.6f}, band_width = {band_width:.4f}")

    # =========================================================================
    # WITNESS CURVES: Compute for each y_dim (reusing same band_width)
    # =========================================================================

    # Effective weights for this x: w_out = C^T k_vec
    w_out = C.T @ k_vec  # (n,)

    witness_curves = {}

    for y_dim, L_grid in context['L_grids'].items():
        # Witness curve: ψ(x, y) = w_out^T L(·, y)
        witness_curve = (w_out @ L_grid).cpu().numpy()
        witness_curves[y_dim] = witness_curve

    diagnostics = {
        'q_hat': q_hat,
        'T_bootstrap': T_bootstrap,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'n': n
    }

    return band_width, witness_curves, diagnostics


# ==============================================================================
# 4. MAIN COMPUTATION FUNCTION
# ==============================================================================

def compute_all_witness_functions(X_1, A_1, Y_1, X_2, A_2, Y_2,
                                   eval_X, selected_individuals,
                                   y_grid, outcome_names,
                                   alpha=0.05, B=1000, reg=0.001):
    """
    Compute witness functions and confidence bands for multiple individuals.

    Nuisances and profile-independent matrices are computed once and shared.
    Each individual gets their own q_hat (slice at their x), computed using the
    full Y kernel and applying uniformly over all y.
    """
    results = {
        'y_grid': y_grid,
        'outcome_names': outcome_names,
        'selected_individuals': selected_individuals,
        'witness_functions': {},
        'band_widths': {},
        'diagnostics': {}
    }

    total_start = time.time()

    # Pool the fitting folds only; eval_X is excluded from bandwidth selection.
    X_all = np.vstack([X_1, X_2])
    Y_all = np.vstack([Y_1, Y_2])

    sigma_x = median_bandwidth(X_all)
    sigma_y = median_bandwidth(Y_all)

    print(f"Global bandwidths: σ_x={sigma_x:.4f}, σ_y={sigma_y:.4f}")

    results['sigma_x'] = sigma_x
    results['sigma_y'] = sigma_y
    results['n'] = len(X_all)

    y_dims = list(range(len(outcome_names)))

    setup_start = time.time()
    context = prepare_witness_context(
        X_1, A_1, Y_1, X_2, A_2, Y_2,
        y_grid, y_dims, sigma_x, sigma_y, reg=reg
    )
    results['shared_setup_time_sec'] = time.time() - setup_start
    print(f"Shared nuisance fitting and matrix setup: {results['shared_setup_time_sec']:.2f}s")

    for ind_idx, individual in enumerate(selected_individuals):
        print(f"\n{'='*50}")
        print(f"Individual {ind_idx+1} (index {individual})")
        print(f"{'='*50}")

        eval_x = eval_X[individual]

        start = time.time()
        band_width, witness_curves, diag = compute_slice_band_and_witness(
            context, eval_x, alpha=alpha, B=B
        )

        # Store results
        results['band_widths'][ind_idx] = band_width
        results['witness_functions'][ind_idx] = witness_curves
        results['diagnostics'][ind_idx] = diag

        print(f"    Total time for individual: {time.time() - start:.2f}s")

    print(f"\nTotal computation time: {time.time() - total_start:.1f}s")

    return results


# ==============================================================================
# 5. PLOTTING
# ==============================================================================

def plot_witness_functions(results, df, outcome_cols):
    """Plot witness functions with confidence bands."""
    y_grid = results['y_grid']
    outcome_names = results['outcome_names']
    selected_individuals = results['selected_individuals']

    n_individuals = len(selected_individuals)
    n_outcomes = len(outcome_names)

    fig, axes = plt.subplots(n_individuals, n_outcomes,
                              figsize=(6*n_outcomes, 5*n_individuals), squeeze=False)

    # Original scale parameters
    orig_mean_y = np.mean(df[outcome_cols].to_numpy(), axis=0)
    orig_std_y = np.std(df[outcome_cols].to_numpy(), axis=0)

    for ind_idx in range(n_individuals):
        band_width = results['band_widths'][ind_idx]

        for outcome_idx, outcome_name in enumerate(outcome_names):
            ax = axes[ind_idx, outcome_idx]

            witness_fn = results['witness_functions'][ind_idx][outcome_idx]

            # Convert to original scale
            y_grid_orig = y_grid * orig_std_y[outcome_idx] + orig_mean_y[outcome_idx]

            # Uniform band (same width for all y)
            lower = witness_fn - band_width
            upper = witness_fn + band_width

            # Plot
            ax.fill_between(y_grid_orig, lower, upper,
                           alpha=0.3, color='blue', label=f"{100*(1-results.get('alpha', 0.05)):g}% CI")
            ax.plot(y_grid_orig, witness_fn, 'b-', lw=2, label='Witness')

            # Highlight significant regions
            significant = (lower > 0) | (upper < 0)
            for i in range(len(y_grid) - 1):
                if significant[i]:
                    ax.fill_between(
                        [y_grid_orig[i], y_grid_orig[i+1]],
                        [lower[i], lower[i+1]],
                        [upper[i], upper[i+1]],
                        alpha=0.3, color='red', lw=0
                    )

            ax.axhline(0, color='k', ls='--', alpha=0.5)
            ax.set_xlabel(outcome_name, fontsize=11)
            ax.set_ylabel('ψ(x, y)', fontsize=11)
            ax.set_title(f'Individual {ind_idx+1}: {outcome_name}', fontsize=12)
            ax.grid(True, alpha=0.3)

            sig_pct = 100 * np.mean(significant)
            ax.text(0.02, 0.98, f'{sig_pct:.1f}% sig\nwidth={band_width:.4f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.tight_layout()
    return fig, axes


# ==============================================================================
# 6. MAIN
# ==============================================================================

def load_pension_data(data_path):
    """Apply the original 401(k) variables, standardization, and held-out split.

    The 1% evaluation subset preserves the profiles in the original witness
    script; the remaining 99% is returned in its original training order.
    Standardization uses the full data, as in that script.
    """
    df = pd.read_csv(data_path)
    treatment_var = "e401"
    outcome_cols = ["net_tfa", "net_nifa", "tw"]
    cont_covariates = ["age", "inc", "fsize", "educ"]
    bin_covariates = ["db", "marr", "twoearn", "pira", "hown"]
    X_cont = standardize_data(df[cont_covariates].to_numpy())
    X_bin = df[bin_covariates].to_numpy()
    X_ = np.concatenate([X_cont, X_bin], axis=1)
    A_ = df[treatment_var].to_numpy()
    Y_ = standardize_data(df[outcome_cols].to_numpy())
    eval_X, X, eval_A, A, eval_Y, Y, eval_rows, train_rows = train_test_split(
        X_, A_, Y_, np.arange(len(df)), test_size=0.99, shuffle=True, random_state=42
    )
    return {
        "frame": df,
        "X": X, "A": A, "Y": Y,
        "eval_X": eval_X, "eval_A": eval_A, "eval_Y": eval_Y,
        "eval_rows": eval_rows, "train_rows": train_rows,
        "outcome_cols": outcome_cols,
        "cont_covariates": cont_covariates,
        "bin_covariates": bin_covariates,
        "treatment_var": treatment_var,
    }
