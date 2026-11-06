"""
Proposed MMD and Wald Test Implementation

This module provides bootstrap-based testing for conditional distributional treatment effects
using the proposed cross-fitted MMD and Wald-type statistics.
"""

import numpy as np
import torch
import time
from typing import Optional, Sequence, Tuple, Literal
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)

# Import utility functions
from .utils import (
    prepare_proposed_mmd_context,
    proposed_mmd_statistic_components_from_context,
    proposed_nystrom_components_from_context,
    wald_intermediate_matrices,
    proposed_fast_mmd_setup,
    proposed_fast_wald_setup,
    proposed_fast_wald_step,
    proposed_nystrom_mmd_setup,
    proposed_nystrom_mmd_step,
    proposed_nystrom_wald_setup,
    proposed_nystrom_wald_step,
    validate_test_inputs,
    positive_finite,
    positive_integer,
    validate_statistics,
    validate_device,
)

def proposed_mmd_wald_test(
    X: NDArray[np.floating],
    A: NDArray[np.floating],
    Y: NDArray[np.floating],
    test_type: Literal["mmd", "wald"] = "mmd",
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    reg: float = 0.001,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    device: Optional[torch.device] = None,
    random_state: Optional[int] = None,
    lgbm_random_state_seed: Optional[int] = None,
    propensity: Optional[NDArray[np.floating]] = None,
    gamma: Optional[float] = None,
    kernel_approximation: Literal["exact", "nystrom"] = "exact",
    nystrom_rank_x: Optional[int] = None,
    nystrom_rank_y: Optional[int] = None,
    nystrom_seed: Optional[int] = None,
    nystrom_solve_mode: Literal["auto", "primal", "dual"] = "auto",
) -> Tuple[float, float, bool, dict]:
    """
    Perform proposed MMD or Wald test for conditional distributional treatment effects with bootstrap.
    
    This function tests the null hypothesis of equality of conditional outcome distributions across treatment groups
    using cross-fitted MMD or Wald-type statistics with bootstrap p-values.
    
    Args:
        X: Covariates, shape (n_samples, n_features)
        A: Treatment assignments, shape (n_samples,) with values in {0, 1}
        Y: Outcomes, shape (n_samples,)
        test_type: Type of test statistic - "mmd" or "wald"
        sigma_x: Bandwidth parameter for covariate kernel
        sigma_y: Bandwidth parameter for outcome kernel
        reg: Regularization parameter for kernel ridge regression
        n_bootstrap: Number of bootstrap samples for p-value computation
        alpha: Significance level for hypothesis test
        device: Torch device; defaults to CUDA when available, otherwise CPU
        random_state: Random seed for bootstrap sampling
        lgbm_random_state_seed: Random seed for LightGBM propensity score models
        propensity: vector of true propensities if known
        kernel_approximation: "exact" for the original full-kernel test, "nystrom" for the
                   Nyström-approximated variant.
        nystrom_rank_x: Nyström rank for the covariate kernel if kernel_approximation="nystrom".
        nystrom_rank_y: Nyström rank for the outcome kernel if kernel_approximation="nystrom".
        nystrom_seed: Random seed for Nyström landmark selection.
        nystrom_solve_mode: "auto", "primal", or "dual" for Nyström Wald solves.
    
    Returns:
        Tuple of (test_statistic, p_value, rejected, info_dict)
        - test_statistic: Scaled test statistic value (n * mmd or n * wald)
        - p_value: Bootstrap-based p-value
        - rejected: Boolean indicating if null hypothesis is rejected at alpha level
        - info_dict: Dictionary with additional information (raw_mmd, eps, etc.)
    
    Raises:
        ValueError: If inputs have incompatible shapes or invalid values
        
    Notes:
        - Uses cross-fitted estimation to reduce overfitting bias
        - Bootstrap weights are generated using multinomial resampling
        - Exact Wald reuses the observed-data covariance and its factorization throughout the bootstrap
    """
    device = validate_device(device)
    total_start_time = time.time()

    # Input validation
    X, A, Y = validate_test_inputs(X, A, Y, propensity)
    positive_finite(reg, "reg")
    positive_integer(n_bootstrap, "n_bootstrap")
    if sigma_x is not None:
        positive_finite(sigma_x, "sigma_x")
    if sigma_y is not None:
        positive_finite(sigma_y, "sigma_y")
    if gamma is not None:
        positive_finite(gamma, "gamma")
    
    if not (len(X) == len(A) == len(Y)):
        raise ValueError(f"Input arrays must have same length: X={len(X)}, A={len(A)}, Y={len(Y)}")
    
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Treatment assignments A must be binary (0 or 1)")
    
    if test_type not in ["mmd", "wald"]:
        raise ValueError(f"test_type must be 'mmd' or 'wald', got {test_type}")

    if kernel_approximation not in ["exact", "nystrom"]:
        raise ValueError(
            f"kernel_approximation must be 'exact' or 'nystrom', got {kernel_approximation}"
        )
    
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    
    n_samples = len(X)
    if n_samples < 4:
        raise ValueError(f"Need at least 4 samples for cross-fitting, got {n_samples}")

    if kernel_approximation == "nystrom":
        if nystrom_rank_x is None or nystrom_rank_y is None:
            raise ValueError("Nyström approximation requires both nystrom_rank_x and nystrom_rank_y")
        positive_integer(nystrom_rank_x, "nystrom_rank_x")
        positive_integer(nystrom_rank_y, "nystrom_rank_y")
        if nystrom_solve_mode not in {"auto", "primal", "dual"}:
            raise ValueError("nystrom_solve_mode must be 'auto', 'primal' or 'dual'")
    
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    context = prepare_proposed_mmd_context(
        X,
        A,
        Y,
        sigma_y=sigma_y,
        device=device,
        lgbm_random_state_seed=lgbm_random_state_seed,
        propensity=propensity,
    )
    pscore_time_sec = float(context.get("pscore_time_sec", 0.0))

    n1 = context["n1"]
    n2 = context["n2"]
    sort_idx1 = context["sort_idx1"]
    sort_idx2 = context["sort_idx2"]

    K = L = None
    nystrom_artifacts = None

    if kernel_approximation == "exact":
        mmd, K, L, _, C, E = proposed_mmd_statistic_components_from_context(
            context,
            sigma_x=sigma_x,
            reg_default=reg,
        )
    else:
        nystrom_artifacts = proposed_nystrom_components_from_context(
            context,
            sigma_x=sigma_x,
            reg_default=reg,
            nystrom_rank_x=int(nystrom_rank_x),
            nystrom_rank_y=int(nystrom_rank_y),
            nystrom_seed=nystrom_seed if nystrom_seed is not None else random_state,
        )
        mmd = nystrom_artifacts["mmd_value"]
        C = nystrom_artifacts["C_torch"]
        E = nystrom_artifacts["E_torch"]
    
    # Compute test statistic
    test_statistic = n_samples * mmd

    eps_used = None
    lu_pivot = None  # Will be used for fast Wald
    bootstrap_ops = None
    D1, D2, V1, V2, W1, W2 = None, None, None, None, None, None 

    if test_type == "mmd":
        if kernel_approximation == "exact":
            MMD_op = proposed_fast_mmd_setup(K, L, C)
            bootstrap_ops = {"MMD_op": MMD_op}
        else:
            bootstrap_ops = proposed_nystrom_mmd_setup(
                nystrom_artifacts["Z_x"],
                nystrom_artifacts["u_rows"],
            )

    elif test_type == "wald":
        if kernel_approximation == "exact":
            D1, D2, V1, V2, W1, W2 = wald_intermediate_matrices(C, E, n1, n2)
            I = torch.eye(2 * n_samples + 4, device=device)

            lu_pivot, eps_used, bootstrap_ops = proposed_fast_wald_setup(
                K, C, L, D1, D2, V1, V2, W1, W2, I, eps=None, gamma=gamma
            )
            ref_tensor = bootstrap_ops["MMD_op"]
            ones_vec = torch.ones(n_samples, dtype=ref_tensor.dtype, device=ref_tensor.device)
            wald = proposed_fast_wald_step(ones_vec, lu_pivot, bootstrap_ops, eps_used)
        else:
            bootstrap_ops, eps_used = proposed_nystrom_wald_setup(
                nystrom_artifacts["Z_x"],
                nystrom_artifacts["Z_y"],
                C,
                E,
                n1,
                n2,
                eps=None,
                gamma=gamma,
                solve_mode=nystrom_solve_mode,
            )
            ones_vec = torch.ones(n_samples, dtype=C.dtype, device=C.device)
            wald = proposed_nystrom_wald_step(ones_vec, bootstrap_ops, eps_used)
        test_statistic = n_samples * wald
    
    # Bootstrap to get p-value
    bootstrap_samples = np.zeros(n_bootstrap)
    
    # logger.info(f"Running {test_type.upper()} test with {n_bootstrap} bootstrap samples...")
    
    for b in range(n_bootstrap):
        if b > 0 and b % 500 == 0:
            logger.debug(f"Bootstrap progress: {b}/{n_bootstrap}")
            
        # Generate bootstrap weights using multinomial resampling
        mult_1 = np.random.multinomial(n1, [1/n1] * n1) - 1
        mult_2 = np.random.multinomial(n2, [1/n2] * n2) - 1
        
        # Apply sort indices to match the sorted data
        mult_1 = torch.tensor(mult_1[sort_idx1], dtype=torch.float32, device=device)
        mult_2 = torch.tensor(mult_2[sort_idx2], dtype=torch.float32, device=device)
        
        # Combine weights
        mult = torch.cat([mult_1, mult_2], dim=0)
        
        if test_type == "mmd":
            if kernel_approximation == "exact":
                temp = bootstrap_ops["MMD_op"] @ mult
                mmd_b = torch.dot(mult, temp).item()
            else:
                mmd_b = proposed_nystrom_mmd_step(mult, bootstrap_ops)
            bootstrap_samples[b] = n_samples * mmd_b
        else:
            if kernel_approximation == "nystrom":
                wald_b = proposed_nystrom_wald_step(mult, bootstrap_ops, eps_used)
                bootstrap_samples[b] = n_samples * wald_b
            else:
                wald_b = proposed_fast_wald_step(mult, lu_pivot, bootstrap_ops, eps_used)
                bootstrap_samples[b] = n_samples * wald_b
    
    # Calculate p-value
    validate_statistics(test_statistic, bootstrap_samples)
    p_value = np.mean(bootstrap_samples >= test_statistic)
    
    # Test decision
    rejected = p_value <= alpha
    
    # Additional information
    info_dict = {
        'raw_mmd': mmd,
        'n1': n1,
        'n2': n2,
        'eps_used': eps_used if test_type == "wald" else None,
        'bootstrap_samples': bootstrap_samples,
        'test_type': test_type,
        'sigma_x': float(context["sigma_x_median"] if sigma_x is None else sigma_x),
        'sigma_y': float(context["sigma_y"]),
        'n_bootstrap': n_bootstrap,
        'kernel_approximation': kernel_approximation,
        'nystrom_rank_x': nystrom_rank_x if kernel_approximation == "nystrom" else None,
        'nystrom_rank_y': nystrom_rank_y if kernel_approximation == "nystrom" else None,
        'nystrom_solve_mode': (
            bootstrap_ops.get("solve_mode") if (kernel_approximation == "nystrom" and test_type == "wald") else None
        ),
        'nystrom_feature_dim': (
            nystrom_artifacts["d"] if kernel_approximation == "nystrom" else None
        ),
        'pscore_time_sec': pscore_time_sec,
        'total_test_time_sec': time.time() - total_start_time,
    }
    info_dict['test_time_excluding_pscore_sec'] = info_dict['total_test_time_sec'] - pscore_time_sec
    
    return test_statistic, p_value, rejected, info_dict


def proposed_mmd_test(
    X: NDArray[np.floating],
    A: NDArray[np.floating], 
    Y: NDArray[np.floating],
    **kwargs
) -> Tuple[float, float, bool, dict]:
    """
    Convenience function for proposed MMD test only.
    
    Args:
        X: Covariates, shape (n_samples, n_features)
        A: Treatment assignments, shape (n_samples,)
        Y: Outcomes, shape (n_samples,)
        **kwargs: Additional arguments passed to proposed_mmd_wald_test
    
    Returns:
        Same as proposed_mmd_wald_test with test_type="mmd"
    """
    return proposed_mmd_wald_test(X, A, Y, test_type="mmd", **kwargs)


def proposed_wald_test(
    X: NDArray[np.floating],
    A: NDArray[np.floating],
    Y: NDArray[np.floating], 
    **kwargs
) -> Tuple[float, float, bool, dict]:
    """
    Convenience function for proposed Wald test only.
    
    Args:
        X: Covariates, shape (n_samples, n_features)
        A: Treatment assignments, shape (n_samples,)
        Y: Outcomes, shape (n_samples,)
        **kwargs: Additional arguments passed to proposed_mmd_wald_test
    
    Returns:
        Same as proposed_mmd_wald_test with test_type="wald"
    """
    return proposed_mmd_wald_test(X, A, Y, test_type="wald", **kwargs)


def proposed_mmd_bandwidth_sweep_test(
    X: NDArray[np.floating],
    A: NDArray[np.floating],
    Y: NDArray[np.floating],
    bandwidth_multipliers: Optional[Sequence[float]] = None,
    sigma_y: Optional[float] = None,
    reg: float = 0.001,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    device: Optional[torch.device] = None,
    random_state: Optional[int] = None,
    lgbm_random_state_seed: Optional[int] = None,
    propensity: Optional[NDArray[np.floating]] = None,
    include_aggregate: bool = True,
    aggregate_mode: str = "standardized_max",
    aggregate_quantile_level: Optional[float] = None,
) -> dict:
    """
    Run a finite bandwidth sweep for SKCD-MMD and optionally calibrate Agg(H_sweep)
    using a shared bootstrap across all bandwidths.
    """
    device = validate_device(device)

    X, A, Y = validate_test_inputs(X, A, Y, propensity)
    positive_finite(reg, "reg")
    positive_integer(n_bootstrap, "n_bootstrap")

    if not (len(X) == len(A) == len(Y)):
        raise ValueError(f"Input arrays must have same length: X={len(X)}, A={len(A)}, Y={len(Y)}")
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Treatment assignments A must be binary (0 or 1)")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(X) < 4:
        raise ValueError(f"Need at least 4 samples for cross-fitting, got {len(X)}")

    if bandwidth_multipliers is None:
        bandwidth_multipliers = (0.25, 0.5, 1.0, 2.0, 4.0)
    bandwidth_multipliers = np.asarray(bandwidth_multipliers, dtype=float)
    if bandwidth_multipliers.ndim != 1 or len(bandwidth_multipliers) == 0:
        raise ValueError("bandwidth_multipliers must be a non-empty 1D sequence")
    if not np.all(np.isfinite(bandwidth_multipliers)) or np.any(bandwidth_multipliers <= 0):
        raise ValueError("bandwidth_multipliers must contain positive finite values")
    if aggregate_mode not in {"raw_max", "standardized_max"}:
        raise ValueError("aggregate_mode must be one of {'raw_max', 'standardized_max'}")
    if aggregate_quantile_level is None:
        aggregate_quantile_level = 1.0 - alpha
    if not (0.0 < aggregate_quantile_level < 1.0):
        raise ValueError("aggregate_quantile_level must lie in (0, 1)")

    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    context = prepare_proposed_mmd_context(
        X,
        A,
        Y,
        sigma_y=sigma_y,
        device=device,
        lgbm_random_state_seed=lgbm_random_state_seed,
        propensity=propensity,
    )

    sigma_x_center = context["sigma_x_median"]
    sigma_x_values = sigma_x_center * bandwidth_multipliers
    n_samples = len(X)

    observed_stats = np.zeros(len(sigma_x_values))
    mmd_ops = []
    bandwidth_results = []

    for idx, sigma_x_value in enumerate(sigma_x_values):
        mmd_value, K_torch, L_torch, _, C_torch, _ = proposed_mmd_statistic_components_from_context(
            context,
            sigma_x=float(sigma_x_value),
            reg_default=reg
        )
        test_statistic = n_samples * mmd_value
        observed_stats[idx] = test_statistic
        mmd_ops.append(proposed_fast_mmd_setup(K_torch, L_torch, C_torch))
        bandwidth_results.append({
            "bandwidth_multiplier": float(bandwidth_multipliers[idx]),
            "sigma_x": float(sigma_x_value),
            "raw_mmd": float(mmd_value),
            "test_statistic": float(test_statistic),
        })

    stacked_mmd_ops = torch.stack(mmd_ops, dim=0)
    bootstrap_samples = np.zeros((n_bootstrap, len(sigma_x_values)))

    for b in range(n_bootstrap):
        if b > 0 and b % 500 == 0:
            logger.debug(f"Bandwidth sweep bootstrap progress: {b}/{n_bootstrap}")

        mult_1 = np.random.multinomial(context["n1"], [1 / context["n1"]] * context["n1"]) - 1
        mult_2 = np.random.multinomial(context["n2"], [1 / context["n2"]] * context["n2"]) - 1

        mult_1 = torch.tensor(mult_1[context["sort_idx1"]], dtype=torch.float32, device=device)
        mult_2 = torch.tensor(mult_2[context["sort_idx2"]], dtype=torch.float32, device=device)
        mult = torch.cat([mult_1, mult_2], dim=0)

        temp = torch.matmul(stacked_mmd_ops, mult)
        boot_stats = n_samples * torch.sum(temp * mult.unsqueeze(0), dim=1)
        boot_stats_np = boot_stats.detach().cpu().numpy()
        bootstrap_samples[b, :] = boot_stats_np

    validate_statistics(observed_stats, bootstrap_samples)
    p_values = np.mean(bootstrap_samples >= observed_stats[None, :], axis=0)

    for idx, result in enumerate(bandwidth_results):
        result["p_value"] = float(p_values[idx])
        result["rejected"] = bool(p_values[idx] <= alpha)

    aggregate_result = None
    aggregate_bootstrap = None
    aggregate_standardizers = None
    if include_aggregate:
        if aggregate_mode == "raw_max":
            aggregate_standardizers = np.ones(len(observed_stats), dtype=float)
        else:
            aggregate_standardizers = np.quantile(
                bootstrap_samples,
                aggregate_quantile_level,
                axis=0,
            )
            if np.any(~np.isfinite(aggregate_standardizers)) or np.any(aggregate_standardizers <= 0):
                raise ValueError(
                    "Marginal bootstrap quantiles used for standardized aggregation must be finite and positive"
                )

        standardized_observed = observed_stats / aggregate_standardizers
        aggregate_bootstrap = np.max(
            bootstrap_samples / aggregate_standardizers[None, :],
            axis=1,
        )
        aggregate_statistic = float(np.max(standardized_observed))
        validate_statistics(aggregate_statistic, aggregate_bootstrap)
        aggregate_p_value = float(np.mean(aggregate_bootstrap >= aggregate_statistic))
        aggregate_result = {
            "test_statistic": aggregate_statistic,
            "p_value": aggregate_p_value,
            "rejected": bool(aggregate_p_value <= alpha),
            "bandwidth_multipliers": [float(x) for x in bandwidth_multipliers],
            "aggregate_mode": aggregate_mode,
            "aggregate_quantile_level": float(aggregate_quantile_level),
            "aggregate_standardizers": [float(x) for x in aggregate_standardizers],
        }

    return {
        "bandwidth_results": bandwidth_results,
        "aggregate_result": aggregate_result,
        "sigma_x_center": float(sigma_x_center),
        "sigma_y": float(context["sigma_y"]),
        "n1": int(context["n1"]),
        "n2": int(context["n2"]),
        "bandwidth_multipliers": [float(x) for x in bandwidth_multipliers],
        "sigma_x_values": [float(x) for x in sigma_x_values],
        "bootstrap_samples": bootstrap_samples,
        "aggregate_bootstrap_samples": aggregate_bootstrap,
        "aggregate_mode": aggregate_mode,
        "aggregate_quantile_level": float(aggregate_quantile_level),
        "aggregate_standardizers": None if aggregate_standardizers is None else [float(x) for x in aggregate_standardizers],
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
    }


def proposed_wald_bandwidth_sweep_test(
    X: NDArray[np.floating],
    A: NDArray[np.floating],
    Y: NDArray[np.floating],
    bandwidth_multipliers: Optional[Sequence[float]] = None,
    sigma_y: Optional[float] = None,
    reg: float = 0.001,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    device: Optional[torch.device] = None,
    random_state: Optional[int] = None,
    lgbm_random_state_seed: Optional[int] = None,
    propensity: Optional[NDArray[np.floating]] = None,
    include_aggregate: bool = True,
    aggregate_mode: str = "standardized_max",
    aggregate_quantile_level: Optional[float] = None,
    gamma: Optional[float] = None,
) -> dict:
    """
    Run a finite bandwidth sweep for SKCD-Wald and optionally calibrate Agg(H_sweep)
    using a shared bootstrap across all bandwidths.
    """
    device = validate_device(device)

    X, A, Y = validate_test_inputs(X, A, Y, propensity)
    positive_finite(reg, "reg")
    positive_integer(n_bootstrap, "n_bootstrap")
    if gamma is not None:
        positive_finite(gamma, "gamma")

    if not (len(X) == len(A) == len(Y)):
        raise ValueError(f"Input arrays must have same length: X={len(X)}, A={len(A)}, Y={len(Y)}")
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Treatment assignments A must be binary (0 or 1)")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(X) < 4:
        raise ValueError(f"Need at least 4 samples for cross-fitting, got {len(X)}")

    if bandwidth_multipliers is None:
        bandwidth_multipliers = (0.25, 0.5, 1.0, 2.0, 4.0)
    bandwidth_multipliers = np.asarray(bandwidth_multipliers, dtype=float)
    if bandwidth_multipliers.ndim != 1 or len(bandwidth_multipliers) == 0:
        raise ValueError("bandwidth_multipliers must be a non-empty 1D sequence")
    if not np.all(np.isfinite(bandwidth_multipliers)) or np.any(bandwidth_multipliers <= 0):
        raise ValueError("bandwidth_multipliers must contain positive finite values")
    if aggregate_mode not in {"raw_max", "standardized_max"}:
        raise ValueError("aggregate_mode must be one of {'raw_max', 'standardized_max'}")
    if aggregate_quantile_level is None:
        aggregate_quantile_level = 1.0 - alpha
    if not (0.0 < aggregate_quantile_level < 1.0):
        raise ValueError("aggregate_quantile_level must lie in (0, 1)")

    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    context = prepare_proposed_mmd_context(
        X,
        A,
        Y,
        sigma_y=sigma_y,
        device=device,
        lgbm_random_state_seed=lgbm_random_state_seed,
        propensity=propensity,
    )

    sigma_x_center = context["sigma_x_median"]
    sigma_x_values = sigma_x_center * bandwidth_multipliers
    n_samples = len(X)
    identity_matrix = torch.eye(2 * n_samples + 4, device=device)

    observed_stats = np.zeros(len(sigma_x_values))
    bandwidth_setups = []
    bandwidth_results = []

    for idx, sigma_x_value in enumerate(sigma_x_values):
        mmd_value, K_torch, L_torch, _, C_torch, E_torch = proposed_mmd_statistic_components_from_context(
            context,
            sigma_x=float(sigma_x_value),
            reg_default=reg,
        )
        D1, D2, V1, V2, W1, W2 = wald_intermediate_matrices(
            C_torch, E_torch, context["n1"], context["n2"]
        )
        lu_pivot, eps_used, bootstrap_ops = proposed_fast_wald_setup(
            K_torch,
            C_torch,
            L_torch,
            D1,
            D2,
            V1,
            V2,
            W1,
            W2,
            identity_matrix,
            eps=None,
            gamma=gamma,
        )

        ref_tensor = bootstrap_ops["MMD_op"]
        ones_vec = torch.ones(n_samples, dtype=ref_tensor.dtype, device=ref_tensor.device)
        wald_value = proposed_fast_wald_step(ones_vec, lu_pivot, bootstrap_ops, eps_used)
        test_statistic = n_samples * wald_value

        observed_stats[idx] = test_statistic
        bandwidth_setups.append((lu_pivot, float(eps_used), bootstrap_ops))
        bandwidth_results.append({
            "bandwidth_multiplier": float(bandwidth_multipliers[idx]),
            "sigma_x": float(sigma_x_value),
            "raw_mmd": float(mmd_value),
            "raw_wald": float(wald_value),
            "eps_used": float(eps_used),
            "test_statistic": float(test_statistic),
        })

    bootstrap_samples = np.zeros((n_bootstrap, len(sigma_x_values)))

    for b in range(n_bootstrap):
        if b > 0 and b % 500 == 0:
            logger.debug(f"Wald bandwidth sweep bootstrap progress: {b}/{n_bootstrap}")

        mult_1 = np.random.multinomial(context["n1"], [1 / context["n1"]] * context["n1"]) - 1
        mult_2 = np.random.multinomial(context["n2"], [1 / context["n2"]] * context["n2"]) - 1

        mult_1 = torch.tensor(mult_1[context["sort_idx1"]], dtype=torch.float32, device=device)
        mult_2 = torch.tensor(mult_2[context["sort_idx2"]], dtype=torch.float32, device=device)
        mult = torch.cat([mult_1, mult_2], dim=0)

        for idx, (lu_pivot, eps_used, bootstrap_ops) in enumerate(bandwidth_setups):
            wald_b = proposed_fast_wald_step(mult, lu_pivot, bootstrap_ops, eps_used)
            bootstrap_samples[b, idx] = n_samples * wald_b

    validate_statistics(observed_stats, bootstrap_samples)
    p_values = np.mean(bootstrap_samples >= observed_stats[None, :], axis=0)

    for idx, result in enumerate(bandwidth_results):
        result["p_value"] = float(p_values[idx])
        result["rejected"] = bool(p_values[idx] <= alpha)

    aggregate_result = None
    aggregate_bootstrap = None
    aggregate_standardizers = None
    if include_aggregate:
        if aggregate_mode == "raw_max":
            aggregate_standardizers = np.ones(len(observed_stats), dtype=float)
        else:
            aggregate_standardizers = np.quantile(
                bootstrap_samples,
                aggregate_quantile_level,
                axis=0,
            )
            if np.any(~np.isfinite(aggregate_standardizers)) or np.any(aggregate_standardizers <= 0):
                raise ValueError(
                    "Marginal bootstrap quantiles used for standardized aggregation must be finite and positive"
                )

        standardized_observed = observed_stats / aggregate_standardizers
        aggregate_bootstrap = np.max(
            bootstrap_samples / aggregate_standardizers[None, :],
            axis=1,
        )
        aggregate_statistic = float(np.max(standardized_observed))
        validate_statistics(aggregate_statistic, aggregate_bootstrap)
        aggregate_p_value = float(np.mean(aggregate_bootstrap >= aggregate_statistic))
        aggregate_result = {
            "test_statistic": aggregate_statistic,
            "p_value": aggregate_p_value,
            "rejected": bool(aggregate_p_value <= alpha),
            "bandwidth_multipliers": [float(x) for x in bandwidth_multipliers],
            "aggregate_mode": aggregate_mode,
            "aggregate_quantile_level": float(aggregate_quantile_level),
            "aggregate_standardizers": [float(x) for x in aggregate_standardizers],
        }

    return {
        "bandwidth_results": bandwidth_results,
        "aggregate_result": aggregate_result,
        "sigma_x_center": float(sigma_x_center),
        "sigma_y": float(context["sigma_y"]),
        "n1": int(context["n1"]),
        "n2": int(context["n2"]),
        "bandwidth_multipliers": [float(x) for x in bandwidth_multipliers],
        "sigma_x_values": [float(x) for x in sigma_x_values],
        "bootstrap_samples": bootstrap_samples,
        "aggregate_bootstrap_samples": aggregate_bootstrap,
        "aggregate_mode": aggregate_mode,
        "aggregate_quantile_level": float(aggregate_quantile_level),
        "aggregate_standardizers": None if aggregate_standardizers is None else [float(x) for x in aggregate_standardizers],
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "gamma": None if gamma is None else float(gamma),
    }
