"""
KCD baseline Test Implementation

This module provides the main test function for KCD baseline statistical testing.
"""

import numpy as np
import torch
from typing import Optional, Tuple
from numpy.typing import NDArray
from .utils import (
    kcd_baseline_statistic, fit_lightgbm_model, validate_device, clip_propensity,
    validate_test_inputs, positive_finite, positive_integer, validate_statistics,
)
def kcd_baseline_test(
    X: NDArray[np.floating], 
    A: NDArray[np.floating], 
    Y: NDArray[np.floating], 
    reg: float = 0.001, 
    num_perm: int = 150, 
    propensity: Optional[NDArray[np.floating]] = None,
    random_state: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Perform KCD baseline test for conditional distributional treatment effects.
    
    This function tests the null hypothesis of equality of conditional outcome distributions across treatment groups
    using the KCD baseline statistic with permutation testing.
    
    Args:
        X: Covariates, shape (n_samples, n_features)
        A: Treatment assignments, shape (n_samples,) with values in {0, 1}
        Y: Outcomes, shape (n_samples,)
        reg: Regularization parameter for kernel ridge regression
        num_perm: Number of permutations for p-value computation
        propensity: Pre-computed propensity scores, shape (n_samples,). 
                   If None, estimates using LightGBM
        random_state: Random seed for reproducible permutations
    
    Returns:
        Tuple of (test_statistic, p_value)
        - test_statistic: KCD baseline statistic value
        - p_value: Permutation-based p-value for the test
    
    Raises:
        ValueError: If inputs have incompatible shapes or invalid values
        
    Notes:
        - The null hypothesis is equality of conditional outcome distributions across treatment groups
        - Small p-values (< 0.05) suggest evidence of a conditional distributional treatment effect
        - Propensity scores are clipped to [1e-6, 1-1e-6] for numerical stability
    """
    device = validate_device(device)

    # Input validation
    X, A, Y = validate_test_inputs(X, A, Y, propensity)
    positive_finite(reg, "reg")
    positive_integer(num_perm, "num_perm")
    
    if not (len(X) == len(A) == len(Y)):
        raise ValueError(f"Input arrays must have same length: X={len(X)}, A={len(A)}, Y={len(Y)}")
    
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Treatment assignments A must be binary (0 or 1)")
    
    if num_perm < 1:
        raise ValueError(f"num_perm must be positive, got {num_perm}")
    
    if not (0 < reg):
        raise ValueError(f"reg must be in > 0, got {reg}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    t_hat = kcd_baseline_statistic(
        X, A, Y, 
        reg_default=reg, 
        device=device
    )

    if propensity is not None:
        if len(propensity) != len(A):
            raise ValueError(f"Propensity length {len(propensity)} != A length {len(A)}")
        e_hat = clip_propensity(propensity)
    else:
        # Fit propensity score model
        model_e_hat = fit_lightgbm_model(X, A)
        e_hat = clip_propensity(model_e_hat.predict(X))
    
    # Permutation test
    t_k = np.zeros(num_perm)
    for k in range(num_perm):
        A_tilde = np.random.binomial(1, e_hat)
        t_k[k] = kcd_baseline_statistic(
            X, A_tilde, Y, 
            reg_default=reg, 
            device=device
        )
    # Compute p-value (including observed statistic in both numerator and denominator)
    validate_statistics(t_hat, t_k)
    p_value = (1 + np.sum(t_k >= t_hat)) / (1 + num_perm)
    
    return t_hat, p_value
