# -*- coding: utf-8 -*-
# @Time       : 2025/9/26 14:54
# @Author     : Jichen Breeze Wen
# E-mail      : Breezewjc952@gmail.com
# @Description: Bootstrap functionality for SCENT

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as rdm
from jaxtyping import Array, ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.stderr import FisherInfoError


def interp_pval(q: ArrayLike) -> float:
    """Description:
        Interpolate a two-sided p-value from null-scaled bootstrap quantiles.

    Args:
        q: Bootstrap quantiles centered around the null.

    Returns:
        Interpolated two-sided p-value.
    """
    R = len(q)
    tstar = jnp.sort(q)
    # Use <= to align with R's findInterval(0, tstar).
    zero = jnp.sum(tstar <= 0)

    if zero == 0 or zero == R:
        return 2.0 / R

    pval = 2.0 * jnp.minimum(zero / R, (R - zero) / R)
    return pval.item()


def basic_p(obs: float, boot: ArrayLike, null: float = 0) -> float:
    """Description:
        Compute the basic bootstrap p-value from observed and bootstrapped estimates.

    Args:
        obs: Observed parameter estimate.
        boot: Bootstrap replicate estimates.
        null: Null hypothesis value.

    Returns:
        Basic bootstrap p-value.
    """
    return interp_pval(2 * obs - boot - null)


def bootstrap_regression(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    bootstrap_indices: ArrayLike,
    atac_idx: int,
) -> Array:
    """Description:
        Fit one bootstrap replicate and extract the ATAC coefficient.

    Args:
        X: Design matrix.
        y: Response vector.
        family: GLM family object.
        bootstrap_indices: Resampled row indices.
        atac_idx: ATAC coefficient index in the fitted beta vector.

    Returns:
        ATAC coefficient from one bootstrap fit.
    """
    X_boot = X[bootstrap_indices]
    y_boot = y[bootstrap_indices]

    glm = GLM(family=family)
    eta, alpha_n = glm.calc_eta_and_dispersion(X_boot, y_boot, jnp.zeros_like(y_boot))
    glm_state = glm.fit(X_boot, y_boot, init=eta, alpha_init=alpha_n, se_estimator=FisherInfoError())

    return glm_state.beta[atac_idx]


def _run_bootstrap_stage(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    key: rdm.PRNGKey,
    atac_idx: int,
    obs_coef: float,
    n_boot: int,
) -> Tuple[float, rdm.PRNGKey]:
    """Description:
        Run one adaptive bootstrap stage with a fixed number of replicates.

    Args:
        X: Design matrix.
        y: Response vector.
        family: GLM family object.
        key: JAX PRNG key.
        atac_idx: ATAC coefficient index.
        obs_coef: Observed ATAC coefficient.
        n_boot: Number of bootstrap replicates for this stage.

    Returns:
        Tuple of stage p-value and advanced PRNG key.
    """
    n = X.shape[0]
    key, subkey = rdm.split(key)
    indices = rdm.choice(subkey, n, shape=(n_boot, n), replace=True)

    boot_coefs = jax.vmap(lambda idx: bootstrap_regression(X, y, family, idx, atac_idx))(indices)
    p0 = basic_p(obs_coef, boot_coefs)

    return p0, key


def bootstrap_test(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    initial_samples: int,
    key: rdm.PRNGKey,
    atac_idx: int = -1,
    obs_coef: Optional[float] = None,
) -> float:
    """Description:
        Perform adaptive bootstrap with stage thresholds aligned to the R implementation.

    Args:
        X: Design matrix.
        y: Response vector.
        family: GLM family object.
        initial_samples: Initial bootstrap sample size (stage-1).
        key: JAX PRNG key.
        atac_idx: ATAC coefficient index.
        obs_coef: Optional observed ATAC coefficient; computed if not provided.

    Returns:
        Final adaptive bootstrap p-value.
    """
    if initial_samples <= 0:
        raise ValueError("initial_samples must be > 0")

    if obs_coef is None:
        glm = GLM(family=family)
        eta, alpha_n = glm.calc_eta_and_dispersion(X, y, jnp.zeros_like(y))
        glm_state = glm.fit(X, y, init=eta, alpha_init=alpha_n, se_estimator=FisherInfoError())
        obs_coef = float(glm_state.beta[atac_idx].item())

    # R path: always run initial stage first, then adaptively rerun with larger R.
    p0, key = _run_bootstrap_stage(X, y, family, key, atac_idx, obs_coef, initial_samples)

    if p0 < 0.1:
        p0, key = _run_bootstrap_stage(X, y, family, key, atac_idx, obs_coef, 500)

    if p0 < 0.05:
        p0, key = _run_bootstrap_stage(X, y, family, key, atac_idx, obs_coef, 2500)

    if p0 < 0.01:
        p0, key = _run_bootstrap_stage(X, y, family, key, atac_idx, obs_coef, 25000)

    if p0 < 0.001:
        p0, key = _run_bootstrap_stage(X, y, family, key, atac_idx, obs_coef, 50000)

    return p0
