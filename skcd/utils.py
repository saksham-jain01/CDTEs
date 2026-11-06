import warnings
import numpy as np
import torch
import math
import time
from typing import Any, Dict, Tuple, Optional, Literal
from numpy.typing import NDArray
import logging
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

try:
    import optuna_integration.lightgbm as lgb
    if not hasattr(lgb, 'LightGBMTuner'):
        try:
            import lightgbm as lgb # Fallback to standard
            HAS_OPTUNA_TUNER = False
        except ImportError:
            lgb = None
            HAS_OPTUNA_TUNER = False
    else:
        HAS_OPTUNA_TUNER = True
except ImportError:
    try:
        import lightgbm as lgb
        HAS_OPTUNA_TUNER = False
    except ImportError:
        lgb = None
        HAS_OPTUNA_TUNER = False

logger = logging.getLogger(__name__)

# Camera-ready Appendix I.1: estimated propensity scores stay in this interval.
PROPENSITY_CLIP = 1e-6


def clip_propensity(values):
    """Apply the paper's numerical-stability bounds to propensity scores."""
    values = np.asarray(values)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Propensity scores must be finite probabilities in [0, 1]")
    return np.clip(values, PROPENSITY_CLIP, 1.0 - PROPENSITY_CLIP)


def validate_test_inputs(X, A, Y, propensity=None):
    """Reject missing/nonfinite observations before kernels or nuisance fitting."""
    X, A, Y = np.asarray(X), np.asarray(A), np.asarray(Y)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[1] == 0 or A.ndim != 1 or Y.ndim not in (1, 2):
        raise ValueError("Expected X of shape (n, d), A of shape (n,), and Y of shape (n,) or (n, q)")
    if Y.ndim == 2 and Y.shape[1] == 0:
        raise ValueError("Y must contain at least one outcome coordinate")
    if not (len(X) == len(A) == len(Y)):
        raise ValueError("X, A and Y must have the same number of observations")
    for name, values in [("X", X), ("A", A), ("Y", Y)]:
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite observations")
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Treatment assignments A must be binary (0 or 1)")
    if propensity is not None:
        propensity = np.asarray(propensity)
        if propensity.shape != (len(A),):
            raise ValueError("Propensity scores must have shape (n,)")
        clip_propensity(propensity)
    return X, A, Y


def positive_finite(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return float(value)


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def validate_statistics(observed, bootstrap):
    """Never convert a failed numerical computation into a small p-value."""
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(bootstrap)):
        raise FloatingPointError("Observed and bootstrap statistics must all be finite")


def _wald_epsilon(trace, eps=None, gamma=None):
    if gamma is not None:
        positive_finite(gamma, "gamma")
    if eps is None:
        positive_finite(trace, "Covariance trace")
        eps = (gamma * trace) / (1 + gamma * trace) if gamma is not None else trace / (3 + trace)
    if not np.isfinite(eps) or not 0 < eps <= 1:
        raise ValueError(f"Wald regularization eps must be finite and in (0, 1], got {eps}")
    return eps

def get_device() -> torch.device:
    """
    Returns the available torch device, prioritizing CUDA.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Validate device availability and fallback to CPU if necessary.
    
    Args:
        device: Torch device to validate. If None, uses get_device()
    
    Returns:
        Valid torch device that is actually available
    """
    if device is None:
        device = get_device()
    
    # Check CUDA availability
    if device.type == 'cuda' and not torch.cuda.is_available():
        logger.warning(f"CUDA device {device} not available. Falling back to CPU.")
        device = torch.device('cpu')
    elif device.type == 'cuda' and torch.cuda.is_available():
        # Check if specific CUDA device index is available
        if device.index is not None and device.index >= torch.cuda.device_count():
            logger.warning(f"CUDA device {device} not available. Using cuda:0 instead.")
            device = torch.device('cuda:0')
    
    # Check MPS (Apple Silicon) availability if specified
    elif device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.warning(f"MPS device not available. Falling back to CPU.")
        device = torch.device('cpu')
    
    return device

def _pairwise_sq_distances(
    X1: NDArray[np.floating], 
    X2: Optional[NDArray[np.floating]] = None, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Helper function to compute pairwise squared distances using torch.
    
    Args:
        X1: First set of points, shape (n_samples, n_features) or (n_samples,)
        X2: Second set of points, shape (m_samples, n_features) or (m_samples,).
            If None, uses X1 for both sets.
        device: Torch device to use. If None, uses get_device()
    
    Returns:
        Pairwise squared distance matrix of shape (n_samples, m_samples)
    """
    device = validate_device(device)
     
    X1 = X1.reshape([-1, 1]) if len(X1.shape) == 1 else X1
    if X2 is None:
        X2 = X1
    else:
        X2 = X2.reshape([-1, 1]) if len(X2.shape) == 1 else X2

    # Convert to torch tensors
    X1_torch = torch.tensor(X1, dtype=torch.float32).to(device)
    X2_torch = torch.tensor(X2, dtype=torch.float32).to(device)

    sq_dists = torch.cdist(X1_torch, X2_torch, p=2).pow(2)
    return sq_dists


def median_bandwidth(
    data: NDArray[np.floating], 
    device: Optional[torch.device] = None
) -> float:
    """
    Calculate median bandwidth for Gaussian kernel.
    
    Args:
        data: Input data array, shape (n_samples, n_features) or (n_samples,)
        device: Torch device for computation. If None, uses get_device()
    
    Returns:
        Median bandwidth as a float scalar
    """
    device = validate_device(device)
    
    data = np.asarray(data)
    if not np.all(np.isfinite(data)):
        raise ValueError("Bandwidth observations must be finite")
    if len(data) < 2:
        return 1.0
    sq_dists_matrix = _pairwise_sq_distances(data, device=device)
    upper_triangle_mask = torch.triu(torch.ones_like(sq_dists_matrix, dtype=bool), diagonal=1)
    pairwise_sq_dists = sq_dists_matrix[upper_triangle_mask]
    median_dist = torch.median(pairwise_sq_dists)
    return _validated_bandwidth(torch.sqrt(median_dist).item())


def gaussian_kernel_gram(
    X1: NDArray[np.floating], 
    X2: Optional[NDArray[np.floating]] = None, 
    sigma: float = 1.0, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Gaussian kernel matrix (RBF kernel).
    
    Args:
        X1: First set of points, shape (n_samples, n_features) or (n_samples,)
        X2: Second set of points, shape (m_samples, n_features) or (m_samples,).
            If None, computes kernel matrix for X1 with itself.
        sigma: RBF kernel bandwidth parameter (standard deviation)
        device: Torch device for computation. If None, uses get_device()
    
    Returns:
        Gaussian kernel matrix of shape (n_samples, m_samples)
    """
    device = validate_device(device)
    sigma = positive_finite(sigma, "sigma")
    sq_dists = _pairwise_sq_distances(X1, X2, device=device)
    return torch.exp(-sq_dists / (2 * (sigma**2)))


def _validated_bandwidth(sigma: Optional[float], fallback: float = 1.0) -> float:
    """
    Use a unit bandwidth for finite, numerically degenerate derived distances.
    """
    if sigma is None or not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"Derived bandwidth must be finite and nonnegative, got {sigma}")
    if sigma <= 1e-6:
        return fallback
    return float(sigma)


# --- Nuisance Estimation - Propensity Score Model ---
def fit_lightgbm_model(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    test_size: float = 0.2,
    objective: str = "binary",
    metric: str = "binary_logloss",
    custom_obj: Optional[callable] = None,
    custom_eval: Optional[callable] = None,
    weights_train: Optional[NDArray[np.floating]] = None,
    weights_val: Optional[NDArray[np.floating]] = None,
    random_state_lgb_split: Optional[int] = None,
    early_stopping_rounds_lgb: int = 10
) -> Any:
    """
    Fits a LightGBM model and returns the booster.
    
    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        test_size: Fraction of data to use for validation
        objective: LightGBM objective function
        metric: LightGBM evaluation metric
        custom_obj: Custom objective function
        custom_eval: Custom evaluation function
        weights_train: Sample weights for training data
        weights_val: Sample weights for validation data
        random_state_lgb_split: Random state for train/validation split
        early_stopping_rounds_lgb: Early stopping rounds
    
    Returns:
        Trained Optuna-tuned LightGBM booster; fitting errors propagate.
    """
    if lgb is None or train_test_split is None or not HAS_OPTUNA_TUNER:
        raise ImportError(
            "Optuna-integrated LightGBM and sklearn are required for paper propensity estimation. "
            "Use oracle propensity or install the missing dependencies."
        )

    X_train = X_train.reshape([-1, 1]) if len(X_train.shape) == 1 else X_train
    y_train = y_train.ravel()

    train_x, val_x, train_y, val_y = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state_lgb_split)

    params = {
        "objective": objective,
        "metric": metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
        # "min_child_samples": 20, # Prevent overfitting to small clusters
        # "reg_lambda": 1.0,       # L2 regularization to prevent hard 0/1 propensities
    }

    train_data = lgb.Dataset(train_x, train_y, weight=weights_train)
    val_data = lgb.Dataset(val_x, val_y, weight=weights_val)
    
    tuner = lgb.LightGBMTuner(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(early_stopping_rounds_lgb, verbose=False)],
        show_progress_bar=False,
        optuna_callbacks=None,
    )
    tuner.run()
    return tuner.get_best_booster()

# --- Nuisance Estimation - Component Matrices (Cross-fitted version) ---
def intermediate_C_and_E_matrices(
    X1: NDArray[np.floating], 
    A1: NDArray[np.floating], 
    X2: NDArray[np.floating], 
    A2: NDArray[np.floating], 
    pi1: NDArray[np.floating], 
    pi2: NDArray[np.floating], 
    K_for_betas: torch.Tensor, 
    reg_default: float = 0.001
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Computes C and E matrices for cross-fitted MMD estimation.
    
    Args:
        X1: Covariates for fold 1, shape (n1, n_features)
        A1: Treatment assignments for fold 1, shape (n1,)
        X2: Covariates for fold 2, shape (n2, n_features)
        A2: Treatment assignments for fold 2, shape (n2,)
        pi1: Propensity scores for fold 1, shape (n1,)
        pi2: Propensity scores for fold 2, shape (n2,)
        K_for_betas: Kernel matrix for regression, shape (n1+n2, n1+n2)
        reg_default: Default regularization parameter
    
    Returns:
        Tuple of (C, E) matrices, both shape (n1+n2, n1+n2)
    """
    n1, n2 = len(X1), len(X2)
    
    n1t, n1c = np.sum(A1), n1 - np.sum(A1)
    n2t, n2c = np.sum(A2), n2 - np.sum(A2)

    # reg = 1 / (n1 + n2) # Adaptive regularization # too aggressive
    reg = reg_default
    # reg = 1.0 / (50*np.sqrt(n1 + n2))  # Adaptive regularization

    pi1 = clip_propensity(pi1)
    pi2 = clip_propensity(pi2)

    ratio_1t = A1 / pi1
    ratio_1c = (1 - A1) / (1 - pi1)
    ratio_2t = A2 / pi2
    ratio_2c = (1 - A2) / (1 - pi2)

    C11 = np.diag((1 / (2 * n1)) * (ratio_1t - ratio_1c))
    C22 = np.diag((1 / (2 * n2)) * (ratio_2t - ratio_2c))

    coef_1t = (1 / (2 * n1)) * (1 - ratio_1t)
    coef_1c = -(1 / (2 * n1)) * (1 - ratio_1c)
    coef_2t = (1 / (2 * n2)) * (1 - ratio_2t)
    coef_2c = -(1 / (2 * n2)) * (1 - ratio_2c)
    
    # Solve for the beta values with regularization
    n1t, n1c, n2t, n2c = int(n1t), int(n1c), int(n2t), int(n2c)
    
    beta_2t = torch.linalg.solve(
        K_for_betas[n1:n1+n2t, n1:n1+n2t] + reg * torch.eye(n2t, device=K_for_betas.device),
        K_for_betas[:n1, n1:n1+n2t].T
    ).T.cpu().numpy()

    beta_2c = torch.linalg.solve(
        K_for_betas[n1+n2t:n1+n2, n1+n2t:n1+n2] + reg * torch.eye(n2c, device=K_for_betas.device),
        K_for_betas[:n1, n1+n2t:].T
    ).T.cpu().numpy()

    beta_1t = torch.linalg.solve(
        K_for_betas[:n1t, :n1t] + reg * torch.eye(n1t, device=K_for_betas.device),
        K_for_betas[n1:, :n1t].T
    ).T.cpu().numpy()

    beta_1c = torch.linalg.solve(
        K_for_betas[n1t:n1, n1t:n1] + reg * torch.eye(n1c, device=K_for_betas.device),
        K_for_betas[n1:, n1t:n1].T
    ).T.cpu().numpy()

    C12 = np.hstack((coef_1t[:, None] * beta_2t, coef_1c[:, None] * beta_2c))
    C21 = np.hstack((coef_2t[:, None] * beta_1t, coef_2c[:, None] * beta_1c))
    
    E11 = np.zeros((n1, n1))
    E22 = np.zeros((n2, n2))
    # Pure outcome contrast: treated minus control in each evaluation fold.
    E12 = (1 / (2 * n1)) * np.hstack((beta_2t, -beta_2c))
    E21 = (1 / (2 * n2)) * np.hstack((beta_1t, -beta_1c))

    C = np.block([[C11, C12], [C21, C22]])
    E = np.block([[E11, E12], [E21, E22]])
    return C, E


def wald_intermediate_matrices(
    C: torch.Tensor, 
    E: torch.Tensor, 
    n1: int, 
    n2: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute intermediate matrices for Wald-type test statistic.
    
    Args:
        C: C matrix from intermediate computation, shape (n1+n2, n1+n2)
        E: E matrix from intermediate computation, shape (n1+n2, n1+n2)
        n1: Number of samples in fold 1
        n2: Number of samples in fold 2
    
    Returns:
        Tuple of (D1, D2, V1, V2, W1, W2) matrices
    """
    D1 = torch.zeros_like(C)
    D1[:n1, :] = math.sqrt(2 * n1) * C[:n1, :]

    D2 = torch.zeros_like(C)
    D2[n1:, :] = math.sqrt(2 * n2) * C[n1:, :]

    V1 = torch.zeros_like(E)
    V1[:n1, :] = math.sqrt(2 / n1) * E[:n1, :]

    V2 = torch.zeros_like(E)
    V2[n1:, :] = math.sqrt(2 / n2) * E[n1:, :]

    W1 = D1 - n1 * V1
    W2 = D2 - n2 * V2
    return D1, D2, V1, V2, W1, W2


def prepare_proposed_mmd_context(
    X: NDArray[np.floating],
    A: NDArray[np.floating],
    Y: NDArray[np.floating],
    sigma_y: Optional[float] = None,
    device: Optional[torch.device] = None,
    lgbm_random_state_seed: Optional[int] = None,
    propensity: Optional[NDArray[np.floating]] = None
) -> Dict[str, Any]:
    """
    Pre-compute the parts of the proposed MMD statistic that are shared across
    different covariate-kernel bandwidths.
    """
    device = validate_device(device)
    X, A, Y = validate_test_inputs(X, A, Y, propensity)
    if sigma_y is not None:
        positive_finite(sigma_y, "sigma_y")

    n = len(X)
    if n < 4:
        raise ValueError(f"Not enough data for 2-fold split (n={n}). Need at least 4.")
    n_half = n // 2

    X1_orig, X2_orig = X[:n_half], X[n_half:]
    A1_orig, A2_orig = A[:n_half], A[n_half:]
    Y1_orig, Y2_orig = Y[:n_half], Y[n_half:]

    sort_idx1 = np.argsort(A1_orig)[::-1]
    X1_sorted, A1_sorted, Y1_sorted = X1_orig[sort_idx1], A1_orig[sort_idx1], Y1_orig[sort_idx1]

    sort_idx2 = np.argsort(A2_orig)[::-1]
    X2_sorted, A2_sorted, Y2_sorted = X2_orig[sort_idx2], A2_orig[sort_idx2], Y2_orig[sort_idx2]

    n1, n2 = len(X1_orig), len(X2_orig)

    X_full_sorted = np.concatenate((X1_sorted, X2_sorted))
    A_full_sorted = np.concatenate((A1_sorted, A2_sorted))
    Y_full_sorted = np.concatenate((Y1_sorted, Y2_sorted))

    sigma_x_median = _validated_bandwidth(median_bandwidth(X_full_sorted, device=device))
    sigma_y = float(sigma_y) if sigma_y is not None else median_bandwidth(Y_full_sorted, device=device)

    if propensity is not None:
        if len(propensity) != n:
            raise ValueError(f"Propensity length {len(propensity)} != data length {n}")

        logger.info("Using provided propensity scores")
        pscore_time_sec = 0.0
        propensity_full = clip_propensity(propensity)
        propensity1_orig = propensity_full[:n_half]
        propensity2_orig = propensity_full[n_half:]
        pi1_sorted = propensity1_orig[sort_idx1]
        pi2_sorted = propensity2_orig[sort_idx2]
    else:
        logger.debug("Estimating propensity scores via cross-fitting")
        pscore_start = time.time()

        X_train_for_pi1, A_train_for_pi1, X_pred_for_pi1 = X2_orig, A2_orig, X1_sorted
        X_train_for_pi2, A_train_for_pi2, X_pred_for_pi2 = X1_orig, A1_orig, X2_sorted

        model_pi1 = fit_lightgbm_model(
            X_train_for_pi1, A_train_for_pi1,
            random_state_lgb_split=lgbm_random_state_seed
        )
        pi1_sorted = model_pi1.predict(X_pred_for_pi1)

        model_pi2 = fit_lightgbm_model(
            X_train_for_pi2, A_train_for_pi2,
            random_state_lgb_split=lgbm_random_state_seed
        )
        pi2_sorted = model_pi2.predict(X_pred_for_pi2)
        pscore_time_sec = time.time() - pscore_start

    sq_dists_x = _pairwise_sq_distances(X_full_sorted, device=device)
    L_torch = gaussian_kernel_gram(Y_full_sorted, sigma=sigma_y, device=device)

    return {
        "device": device,
        "n": n,
        "n1": n1,
        "n2": n2,
        "sort_idx1": sort_idx1,
        "sort_idx2": sort_idx2,
        "X1_sorted": X1_sorted,
        "A1_sorted": A1_sorted,
        "Y1_sorted": Y1_sorted,
        "X2_sorted": X2_sorted,
        "A2_sorted": A2_sorted,
        "Y2_sorted": Y2_sorted,
        "X_full_sorted": X_full_sorted,
        "A_full_sorted": A_full_sorted,
        "Y_full_sorted": Y_full_sorted,
        "pi1_sorted": pi1_sorted,
        "pi2_sorted": pi2_sorted,
        "pscore_time_sec": pscore_time_sec,
        "sigma_x_median": sigma_x_median,
        "sigma_y": sigma_y,
        "sq_dists_x": sq_dists_x,
        "L_torch": L_torch,
    }


def proposed_mmd_statistic_components_from_context(
    context: Dict[str, Any],
    sigma_x: Optional[float] = None,
    reg_default: float = 0.001
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the bandwidth-specific MMD statistic components from a shared context.
    """
    core = proposed_exact_core_components_from_context(
        context,
        sigma_x=sigma_x,
        reg_default=reg_default,
    )
    K_torch = core["K_torch"]
    L_torch = core["L_torch"]
    C_torch = core["C_torch"]
    E_torch = core["E_torch"]

    KCL_torch = K_torch @ C_torch @ L_torch
    mmd_value = torch.sum(C_torch * KCL_torch).item()

    return mmd_value, K_torch, L_torch, KCL_torch, C_torch, E_torch


def proposed_exact_core_components_from_context(
    context: Dict[str, Any],
    sigma_x: Optional[float] = None,
    reg_default: float = 0.001,
) -> Dict[str, torch.Tensor]:
    """
    Build the exact-kernel backend and nuisance-dependent coefficient matrices.
    """
    device = context["device"]
    positive_finite(reg_default, "reg")
    sigma_x = context["sigma_x_median"] if sigma_x is None else positive_finite(sigma_x, "sigma_x")

    K_torch = torch.exp(-context["sq_dists_x"] / (2 * (sigma_x ** 2)))

    C, E = intermediate_C_and_E_matrices(
        context["X1_sorted"],
        context["A1_sorted"],
        context["X2_sorted"],
        context["A2_sorted"],
        context["pi1_sorted"],
        context["pi2_sorted"],
        K_torch,
        reg_default
    )

    C_torch = torch.tensor(C, dtype=torch.float32).to(device)
    E_torch = torch.tensor(E, dtype=torch.float32).to(device)
    L_torch = context["L_torch"]

    return {
        "K_torch": K_torch,
        "L_torch": L_torch,
        "C_torch": C_torch,
        "E_torch": E_torch,
    }


def nystrom_features(
    X: NDArray[np.floating],
    sigma: float,
    rank: int,
    device: Optional[torch.device] = None,
    random_state: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Nyström features Z so that K(X, X) ~= Z Z^T.
    """
    device = validate_device(device)
    # Build the landmark Gram square root in float64 for numerical stability.
    # This is especially important when rank=n, where we want the Nyström path
    # to reproduce the full dense kernel to numerical precision.
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]

    X_torch = torch.tensor(X_arr, dtype=torch.float64, device=device)
    n = X_torch.shape[0]
    rank = max(1, min(int(rank), n))

    if rank == n:
        landmark_idx = torch.arange(n, device=device)
    else:
        rng = np.random.default_rng(random_state)
        landmark_idx = torch.tensor(
            np.sort(rng.choice(n, size=rank, replace=False)),
            dtype=torch.long,
            device=device,
        )

    X_land = X_torch[landmark_idx]
    sq_nm = torch.cdist(X_torch, X_land, p=2).pow(2)
    sq_mm = torch.cdist(X_land, X_land, p=2).pow(2)
    c_nm = torch.exp(-sq_nm / (2 * (sigma ** 2)))
    w_mm = torch.exp(-sq_mm / (2 * (sigma ** 2)))
    evals, evecs = torch.linalg.eigh(0.5 * (w_mm + w_mm.T))
    evals = torch.clamp(evals, min=1e-10)
    w_inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.T
    return_dtype = torch.float64 if rank == n else torch.float32
    return (c_nm @ w_inv_sqrt).to(dtype=return_dtype)


def nystrom_hat_matrices(
    z_x: torch.Tensor,
    a1: NDArray[np.floating],
    a2: NDArray[np.floating],
    reg_default: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cross-fitted KRR hat matrices induced by Nyström features.
    """
    device = z_x.device
    dtype = z_x.dtype
    n1 = len(a1)
    z1 = z_x[:n1]
    z2 = z_x[n1:]
    n1t = int(np.sum(a1))
    n2t = int(np.sum(a2))

    def hat(eval_z: torch.Tensor, train_z: torch.Tensor) -> torch.Tensor:
        gram = train_z.T @ train_z + reg_default * torch.eye(train_z.shape[1], dtype=dtype, device=device)
        return eval_z @ torch.linalg.solve(gram, train_z.T)

    h12_t = hat(z1, z2[:n2t])
    h12_c = hat(z1, z2[n2t:])
    h21_t = hat(z2, z1[:n1t])
    h21_c = hat(z2, z1[n1t:])
    return h12_t, h12_c, h21_t, h21_c


def intermediate_C_and_E_matrices_nystrom(
    A1: NDArray[np.floating],
    A2: NDArray[np.floating],
    pi1: NDArray[np.floating],
    pi2: NDArray[np.floating],
    z_x_for_betas: torch.Tensor,
    reg_default: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute C and E matrices using Nyström-induced KRR hat matrices.
    """
    device = z_x_for_betas.device
    dtype = z_x_for_betas.dtype
    n1, n2 = len(A1), len(A2)

    a1_t = torch.tensor(np.asarray(A1, dtype=np.float32), dtype=dtype, device=device)
    a2_t = torch.tensor(np.asarray(A2, dtype=np.float32), dtype=dtype, device=device)
    pi1_t = torch.tensor(clip_propensity(np.asarray(pi1, dtype=np.float32)), dtype=dtype, device=device)
    pi2_t = torch.tensor(clip_propensity(np.asarray(pi2, dtype=np.float32)), dtype=dtype, device=device)

    n1t, n1c = int(np.sum(A1)), n1 - int(np.sum(A1))
    n2t, n2c = int(np.sum(A2)), n2 - int(np.sum(A2))
    h12_t, h12_c, h21_t, h21_c = nystrom_hat_matrices(z_x_for_betas, A1, A2, reg_default)

    ratio_1t = a1_t / pi1_t
    ratio_1c = (1.0 - a1_t) / (1.0 - pi1_t)
    ratio_2t = a2_t / pi2_t
    ratio_2c = (1.0 - a2_t) / (1.0 - pi2_t)

    c11 = torch.diag((ratio_1t - ratio_1c) / (2.0 * n1))
    c22 = torch.diag((ratio_2t - ratio_2c) / (2.0 * n2))
    c12 = torch.cat([
        ((1.0 - ratio_1t) / (2.0 * n1))[:, None] * h12_t,
        (-(1.0 - ratio_1c) / (2.0 * n1))[:, None] * h12_c
    ], dim=1)
    c21 = torch.cat([
        ((1.0 - ratio_2t) / (2.0 * n2))[:, None] * h21_t,
        (-(1.0 - ratio_2c) / (2.0 * n2))[:, None] * h21_c
    ], dim=1)

    e11 = torch.zeros((n1, n1), dtype=dtype, device=device)
    e22 = torch.zeros((n2, n2), dtype=dtype, device=device)
    e12 = torch.cat([h12_t, -h12_c], dim=1) / (2.0 * n1)
    e21 = torch.cat([h21_t, -h21_c], dim=1) / (2.0 * n2)

    c = torch.cat([torch.cat([c11, c12], dim=1), torch.cat([c21, c22], dim=1)], dim=0)
    e = torch.cat([torch.cat([e11, e12], dim=1), torch.cat([e21, e22], dim=1)], dim=0)
    return c, e


def proposed_nystrom_components_from_context(
    context: Dict[str, Any],
    sigma_x: Optional[float] = None,
    reg_default: float = 0.001,
    nystrom_rank_x: int = 8,
    nystrom_rank_y: int = 8,
    nystrom_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build Nyström-approximated C/E, MMD witness, and feature artifacts from a shared context.
    """
    artifacts = proposed_nystrom_core_components_from_context(
        context,
        sigma_x=sigma_x,
        reg_default=reg_default,
        nystrom_rank_x=nystrom_rank_x,
        nystrom_rank_y=nystrom_rank_y,
        nystrom_seed=nystrom_seed,
    )
    z_x_stat = artifacts["Z_x"]
    z_y = artifacts["Z_y"]
    c_torch = artifacts["C_torch"]

    u_rows = c_torch @ z_y
    witness_matrix = z_x_stat.T @ u_rows
    mmd_value = torch.sum(witness_matrix * witness_matrix).item()

    artifacts.update({
        "mmd_value": mmd_value,
        "u_rows": u_rows,
        "witness_matrix": witness_matrix,
    })
    return artifacts


def proposed_nystrom_core_components_from_context(
    context: Dict[str, Any],
    sigma_x: Optional[float] = None,
    reg_default: float = 0.001,
    nystrom_rank_x: int = 8,
    nystrom_rank_y: int = 8,
    nystrom_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build the Nyström backend and nuisance-dependent coefficient matrices.
    """
    positive_finite(reg_default, "reg")
    sigma_x = context["sigma_x_median"] if sigma_x is None else positive_finite(sigma_x, "sigma_x")
    sigma_y = context["sigma_y"]
    device = context["device"]

    z_x_stat = nystrom_features(
        context["X_full_sorted"],
        sigma=sigma_x,
        rank=nystrom_rank_x,
        device=device,
        random_state=nystrom_seed,
    )

    z_y = nystrom_features(
        context["Y_full_sorted"],
        sigma=sigma_y,
        rank=nystrom_rank_y,
        device=device,
        random_state=None if nystrom_seed is None else nystrom_seed + 1,
    )

    # Full-rank factors use float64; partial factors normally use float32.
    # A mixed-rank pair must share a dtype before constructing coefficients.
    dtype = torch.promote_types(z_x_stat.dtype, z_y.dtype)
    z_x_stat = z_x_stat.to(dtype=dtype)
    z_y = z_y.to(dtype=dtype)

    c_torch, e_torch = intermediate_C_and_E_matrices_nystrom(
        context["A1_sorted"],
        context["A2_sorted"],
        context["pi1_sorted"],
        context["pi2_sorted"],
        z_x_stat,
        reg_default=reg_default,
    )

    return {
        "C_torch": c_torch,
        "E_torch": e_torch,
        "Z_x": z_x_stat,
        "Z_y": z_y,
        "d": int(nystrom_rank_x * nystrom_rank_y),
        "rank_x": int(nystrom_rank_x),
        "rank_y": int(nystrom_rank_y),
    }


def proposed_nystrom_mmd_setup(
    z_x: torch.Tensor,
    u_rows: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Pre-compute Nyström MMD bootstrap operators.
    """
    return {"Z_x": z_x, "u_rows": u_rows}


def proposed_nystrom_mmd_step(
    xi: torch.Tensor,
    ops: Dict[str, torch.Tensor],
) -> float:
    """
    Compute a Nyström MMD bootstrap replicate in O(n d).
    """
    weighted_zx = ops["Z_x"] * xi[:, None]
    w_b = (weighted_zx.T @ ops["u_rows"]).reshape(-1)
    return torch.dot(w_b, w_b).item()


def proposed_nystrom_wald_setup(
    z_x: torch.Tensor,
    z_y: torch.Tensor,
    c_torch: torch.Tensor,
    e_torch: torch.Tensor,
    n1: int,
    n2: int,
    eps: Optional[float] = None,
    gamma: Optional[float] = None,
    solve_mode: Literal["auto", "primal", "dual"] = "auto",
) -> Tuple[Dict[str, Any], float]:
    """
    Pre-compute Nyström Wald operators in finite feature space.

    In the primal regime, avoid materializing the dual Gram matrix F^T F.
    """
    device = z_x.device
    dtype = z_x.dtype
    d1, d2, v1, v2, _, _ = wald_intermediate_matrices(c_torch, e_torch, n1, n2)

    p1 = d1 @ z_y
    p2 = d2 @ z_y
    v1_feat = (z_x.T @ v1 @ z_y).reshape(-1)
    v2_feat = (z_x.T @ v2 @ z_y).reshape(-1)

    d = z_x.shape[1] * z_y.shape[1]
    f1 = (z_x[:n1, :, None] * p1[:n1, None, :]).reshape(n1, d).T - v1_feat[:, None]
    f2 = (z_x[n1:, :, None] * p2[n1:, None, :]).reshape(n2, d).T - v2_feat[:, None]
    f = torch.cat([f1, f2], dim=1)

    # tr(F^T F) = ||F||_F^2; no dual Gram is needed for epsilon selection.
    eps = _wald_epsilon(torch.sum(f * f).item(), eps, gamma)

    if solve_mode not in {"auto", "primal", "dual"}:
        raise ValueError("solve_mode must be 'auto', 'primal' or 'dual'")

    if solve_mode == "auto":
        solve_mode = "primal" if d <= (n1 + n2) else "dual"

    if solve_mode == "primal":
        operator = (eps * torch.eye(d, dtype=dtype, device=device)) + ((1.0 - eps) * (f @ f.T))
        ops = {
            "Z_x": z_x,
            "u_rows": c_torch @ z_y,
            "chol": torch.linalg.cholesky(operator),
            "solve_mode": solve_mode,
        }
    else:
        k_g = f.T @ f
        operator = (eps * torch.eye(n1 + n2, dtype=dtype, device=device)) + ((1.0 - eps) * k_g)
        ops = {
            "Z_x": z_x,
            "u_rows": c_torch @ z_y,
            "F": f,
            "K_g": k_g,
            "chol": torch.linalg.cholesky(operator),
            "solve_mode": solve_mode,
        }
    return ops, eps


def proposed_nystrom_wald_step(
    xi: torch.Tensor,
    ops: Dict[str, Any],
    eps: float,
) -> float:
    """
    Compute a Nyström Wald bootstrap replicate in O(n d + d^2) in primal mode
    and O(n d + n^2) in dual mode.
    """
    weighted_zx = ops["Z_x"] * xi[:, None]
    w_b = (weighted_zx.T @ ops["u_rows"]).reshape(-1)

    if ops["solve_mode"] == "primal":
        z = torch.cholesky_solve(w_b.unsqueeze(1), ops["chol"]).squeeze(1)
        return torch.dot(w_b, z).item()

    a_b = ops["F"].T @ w_b
    z = torch.cholesky_solve(a_b.unsqueeze(1), ops["chol"]).squeeze(1)
    return (((w_b @ w_b) / eps) - (((1.0 - eps) / eps) * (a_b @ z))).item()


def proposed_fast_mmd_setup(
    K: torch.Tensor,
    L: torch.Tensor,
    C: torch.Tensor
) -> torch.Tensor:
    """
    Pre-compute the operator for fast O(n^2) MMD bootstrap.
    
    Returns Phi such that MMD_b = xi.T @ Phi @ xi
    """
    # MMD = Tr( diag(xi) K diag(xi) C L C.T )
    #     = xi.T @ (K * (C L C.T)) @ xi  [Element-wise product of K and CLC^T]
    
    # This is O(n^3) but performed only once.
    CLC_T = C @ L @ C.T
    
    # The operator is the element-wise product of K and CLC_T
    # Note: K is symmetric, CLC_T is symmetric (as L is symmetric).
    MMD_op = K * CLC_T
    
    return MMD_op


def _assemble_wald_ut_t(
    K: torch.Tensor,
    L: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    V1: torch.Tensor,
    V2: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    """
    Assemble the finite-dimensional covariance proxy M = U^T T used by the Wald statistic.
    """
    LD1T = L @ D1.T
    LD2T = L @ D2.T

    S1TGS1 = K * (D1 @ LD1T)
    S1TGS2 = K * (D1 @ LD2T)
    S2TGS1 = K * (D2 @ LD1T)
    S2TGS2 = K * (D2 @ LD2T)

    UT_T_top_left = torch.cat([
        torch.cat([S1TGS1, S1TGS2], dim=1),
        torch.cat([S2TGS1, S2TGS2], dim=1)
    ], dim=0)

    KD1L = K @ D1 @ L
    KD2L = K @ D2 @ L
    S1T_G_d1 = torch.sum(D1 * KD1L, dim=1, keepdim=True)
    S1T_G_d2 = torch.sum(D1 * KD2L, dim=1, keepdim=True)
    S2T_G_d1 = torch.sum(D2 * KD1L, dim=1, keepdim=True)
    S2T_G_d2 = torch.sum(D2 * KD2L, dim=1, keepdim=True)

    KV1L = K @ V1 @ L
    KV2L = K @ V2 @ L
    S1T_G_v1 = torch.sum(D1 * KV1L, dim=1, keepdim=True)
    S1T_G_v2 = torch.sum(D1 * KV2L, dim=1, keepdim=True)
    S2T_G_v1 = torch.sum(D2 * KV1L, dim=1, keepdim=True)
    S2T_G_v2 = torch.sum(D2 * KV2L, dim=1, keepdim=True)

    KW1L = K @ W1 @ L
    KW2L = K @ W2 @ L
    S1T_G_w1 = torch.sum(D1 * KW1L, dim=1, keepdim=True)
    S1T_G_w2 = torch.sum(D1 * KW2L, dim=1, keepdim=True)
    S2T_G_w1 = torch.sum(D2 * KW1L, dim=1, keepdim=True)
    S2T_G_w2 = torch.sum(D2 * KW2L, dim=1, keepdim=True)

    UT_T_top_right = torch.cat([
        torch.cat([S1T_G_v1, S2T_G_v1], dim=0),
        torch.cat([S1T_G_v2, S2T_G_v2], dim=0),
        torch.cat([S1T_G_w1, S2T_G_w1], dim=0),
        torch.cat([S1T_G_w2, S2T_G_w2], dim=0)
    ], dim=1)

    UT_T_bottom_left = torch.cat([
        torch.cat([-S1T_G_d1, -S2T_G_d1], dim=0),
        torch.cat([-S1T_G_d2, -S2T_G_d2], dim=0),
        torch.cat([-S1T_G_v1, -S2T_G_v1], dim=0),
        torch.cat([-S1T_G_v2, -S2T_G_v2], dim=0)
    ], dim=1).T

    D1_KV1L_frob = torch.sum(D1 * KV1L)
    D2_KV1L_frob = torch.sum(D2 * KV1L)
    V1_KV1L_frob = torch.sum(V1 * KV1L)
    V2_KV1L_frob = torch.sum(V2 * KV1L)
    D1_KV2L_frob = torch.sum(D1 * KV2L)
    D2_KV2L_frob = torch.sum(D2 * KV2L)
    V1_KV2L_frob = torch.sum(V1 * KV2L)
    V2_KV2L_frob = torch.sum(V2 * KV2L)
    D1_KW1L_frob = torch.sum(D1 * KW1L)
    D2_KW1L_frob = torch.sum(D2 * KW1L)
    V1_KW1L_frob = torch.sum(V1 * KW1L)
    V2_KW1L_frob = torch.sum(V2 * KW1L)
    D1_KW2L_frob = torch.sum(D1 * KW2L)
    D2_KW2L_frob = torch.sum(D2 * KW2L)
    V1_KW2L_frob = torch.sum(V1 * KW2L)
    V2_KW2L_frob = torch.sum(V2 * KW2L)

    UT_T_bottom_right = torch.cat([
        torch.stack([-D1_KV1L_frob, -D1_KV2L_frob, -D1_KW1L_frob, -D1_KW2L_frob]).unsqueeze(0),
        torch.stack([-D2_KV1L_frob, -D2_KV2L_frob, -D2_KW1L_frob, -D2_KW2L_frob]).unsqueeze(0),
        torch.stack([-V1_KV1L_frob, -V1_KV2L_frob, -V1_KW1L_frob, -V1_KW2L_frob]).unsqueeze(0),
        torch.stack([-V2_KV1L_frob, -V2_KV2L_frob, -V2_KW1L_frob, -V2_KW2L_frob]).unsqueeze(0)
    ], dim=0)

    return torch.cat([
        torch.cat([UT_T_top_left, UT_T_top_right], dim=1),
        torch.cat([UT_T_bottom_left, UT_T_bottom_right], dim=1)
    ], dim=0)

def proposed_fast_wald_setup(
    K: torch.Tensor,
    C: torch.Tensor, # Added C to signature
    L: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    V1: torch.Tensor,
    V2: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    I: torch.Tensor,
    eps: Optional[float] = None,
    gamma: Optional[float] = None
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float, dict]:
    """
    Pre-compute fixed covariance structure, LU factorization, and projection operators
    for fast Wald bootstrap.
    """
    # --- 1. Compute Fixed Covariance (Same as original slow setup) ---
    UT_T = _assemble_wald_ut_t(K, L, D1, D2, V1, V2, W1, W2)

    eps = _wald_epsilon(torch.trace(UT_T).item(), eps, gamma)

    operator_matrix = (eps * I) + ((1 - eps) * UT_T)
    lu_pivot = torch.linalg.lu_factor(operator_matrix)

    # --- 2. Compute Bootstrap Operators (The Fast Part) ---
    
    # MMD Operator
    MMD_op = proposed_fast_mmd_setup(K, L, C)

    # Common Matrix M = C @ L
    M = C @ L
    
    # Vector Projection Operators (Phi_vec @ xi -> vector)
    # Replaces: sum(D * KCL, dim=1)
    # Formula: K * (D @ M.T)
    H_d1 = K * (D1 @ M.T)
    H_d2 = K * (D2 @ M.T)

    # Scalar Projection Operators (phi_vec @ xi -> scalar)
    # Replaces: sum(V * KCL)
    # Formula: row_sums( (M @ V.T) * K )
    def get_scalar_proj_vec(Z):
        # returns vector h such that h.dot(xi) = sum(Z * K(diag(xi)C)L)
        term1 = M @ Z.T
        return torch.sum(term1 * K, dim=1)
    
    h_v1 = get_scalar_proj_vec(V1)
    h_v2 = get_scalar_proj_vec(V2)
    h_w1 = get_scalar_proj_vec(W1)
    h_w2 = get_scalar_proj_vec(W2)

    bootstrap_ops = {
        "MMD_op": MMD_op,
        "H_d1": H_d1, "H_d2": H_d2,
        "h_v1": h_v1, "h_v2": h_v2,
        "h_w1": h_w1, "h_w2": h_w2
    }

    return lu_pivot, eps, bootstrap_ops

def proposed_fast_wald_step(
    xi: torch.Tensor,
    lu_pivot: Tuple[torch.Tensor, torch.Tensor],
    ops: dict,
    eps: float
) -> float:
    """
    Compute Wald statistic in O(n^2) using precomputed operators.
    """
    # 1. Compute MMD efficiently O(n^2)
    # MMD = xi^T @ MMD_op @ xi
    # Note: If xi is just ones (original stat), this recovers original MMD^2
    temp_mmd = ops["MMD_op"] @ xi
    mmd_val = torch.dot(xi, temp_mmd).item()

    # 2. Compute Vector Projections O(n^2)
    S1T_G_c = ops["H_d1"] @ xi
    S2T_G_c = ops["H_d2"] @ xi

    # 3. Compute Scalar Projections O(n)
    V1_KCL_frob = torch.dot(ops["h_v1"], xi)
    V2_KCL_frob = torch.dot(ops["h_v2"], xi)
    W1_KCL_frob = torch.dot(ops["h_w1"], xi)
    W2_KCL_frob = torch.dot(ops["h_w2"], xi)

    # Assemble RHS vectors
    cT_T = torch.cat((
        S1T_G_c,
        S2T_G_c,
        torch.stack([V1_KCL_frob, V2_KCL_frob, W1_KCL_frob, W2_KCL_frob])
    ))

    UT_G_c = torch.cat((
        S1T_G_c,
        S2T_G_c,
        torch.stack([-torch.sum(S1T_G_c), -torch.sum(S2T_G_c), 
                     -V1_KCL_frob, -V2_KCL_frob])
    ))
    
    # Note on UT_G_c bottom part: 
    # In original code: D1_KCL_frob = torch.sum(S1T_G_c).
    # Since we computed S1T_G_c explicitly above, we can just sum it.
    
    # Solve linear system O(n^2)
    z = torch.linalg.lu_solve(lu_pivot[0], lu_pivot[1], UT_G_c.unsqueeze(1)).squeeze()

    # Final result
    term2 = ((1 - eps) / eps) * torch.dot(cT_T, z)
    term1 = (1 / eps) * mmd_val

    return (term1 - term2).item()


def kcd_baseline_statistic(
    X: NDArray[np.floating], 
    A: NDArray[np.floating], 
    Y: NDArray[np.floating], 
    reg_default: float = 0.001, 
    device: Optional[torch.device] = None
) -> float:
    """
    Compute KCD baseline statistic components.
    
    Args:
        X: Covariates, shape (n_samples, n_features)
        A: Treatment assignments, shape (n_samples,)
        Y: Outcomes, shape (n_samples,)
        reg_default: Default regularization parameter
        device: Torch device for computation. If None, uses get_device()
    
    Returns:
        KCD baseline statistic value
    """
    device = validate_device(device)

    n = len(X)
    nt = np.sum(A).astype(int)
    nc = n - nt

    sort_idx = np.argsort(A)[::-1] # Get indices for descending sort
    X_sorted, A_sorted, Y_sorted = X[sort_idx], A[sort_idx], Y[sort_idx]

    sigma_y = median_bandwidth(Y_sorted, device=device)
    L = gaussian_kernel_gram(Y_sorted, sigma=sigma_y, device=device)

    Xt = X_sorted[:nt, :]
    Xc = X_sorted[nt:, :]

    sigma_xt = median_bandwidth(Xt, device=device)
    sigma_xc = median_bandwidth(Xc, device=device)

    K_X_Xt = gaussian_kernel_gram(X_sorted, Xt, sigma=sigma_xt, device=device) # This is K(X_eval, X_t)
    K_X_Xc = gaussian_kernel_gram(X_sorted, Xc, sigma=sigma_xc, device=device) # This is K(X_eval, X_c)

    # reg_t = 1/nt # Adaptive regularization # too aggressive
    # reg_c = 1/nc # Adaptive regularization # too aggressive
    reg_t = reg_default
    reg_c = reg_default

    # reg_t = 1.0 / (50*np.sqrt(nt)) # Adaptive regularization
    # reg_c = 1.0 / (50*np.sqrt(nc)) # Adaptive regularization

    Mt = torch.linalg.solve(K_X_Xt[:nt, :] + reg_t * torch.eye(nt, device=device),
                            K_X_Xt.T).T
    Mc = torch.linalg.solve(K_X_Xc[nt:, :] + reg_c * torch.eye(nc, device=device),
                            K_X_Xc.T).T # (N_eval_points, n_c)

    term1 = torch.trace(Mc @ L[nt:, nt:] @ Mc.T)
    term2 = torch.trace(Mc @ L[nt:, :nt] @ Mt.T)
    term3 = torch.trace(Mt @ L[:nt, :nt] @ Mt.T)

    return ((term1 - 2 * term2 + term3) / n).item()
