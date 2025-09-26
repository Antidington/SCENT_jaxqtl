# -*- coding: utf-8 -*-
# @Time       : 2025/9/26 14:54
# @Author     : Jichen Breeze Wen
# E-mail      : Breezewjc952@gmail.com
# @Description: Bootstrap functionality for SCENT

import jax
import jax.numpy as jnp
import jax.random as rdm
from jaxtyping import Array, ArrayLike

from jaxqtl.families.distribution import ExponentialFamily
from jaxqtl.infer.glm import GLM
from jaxqtl.infer.stderr import FisherInfoError


def interp_pval(q: ArrayLike) -> float:
    """Interpolate a p-value from quantiles that should be "null scaled"
    
    Args:
        q: Bootstrap quantiles, centered so that under the null, theta = 0
        
    Returns:
        Two-sided p-value
    """
    R = len(q)
    tstar = jnp.sort(q)
    zero = jnp.sum(tstar < 0)
    
    # Handle extreme cases
    if zero == 0 or zero == R:
        return 2.0 / R
    
    pval = 2.0 * jnp.minimum(zero / R, (R - zero) / R)
    return pval.item()


def basic_p(obs: float, boot: ArrayLike, null: float = 0) -> float:
    """Derive a p-value from a vector of bootstrap samples using the "basic" calculation
    
    Args:
        obs: Observed value of parameter (using actual data)
        boot: Vector of bootstraps
        null: Null hypothesis value
        
    Returns:
        p-value
    """
    return interp_pval(2 * obs - boot - null)


def bootstrap_regression(
    X: ArrayLike, 
    y: ArrayLike, 
    family: ExponentialFamily,
    bootstrap_indices: ArrayLike,
    atac_idx: int = -1
) -> Array:
    """Perform regression on bootstrapped data
    
    Args:
        X: Design matrix
        y: Response variable
        family: GLM family
        bootstrap_indices: Indices for bootstrapping
        atac_idx: Index of ATAC in the design matrix
        
    Returns:
        Coefficient of ATAC effect
    """
    # Get bootstrapped data
    X_boot = X[bootstrap_indices]
    y_boot = y[bootstrap_indices]
    
    # Fit model
    glm = GLM(family=family)
    eta, alpha_n = glm.calc_eta_and_dispersion(X_boot, y_boot, jnp.zeros_like(y_boot))
    glm_state = glm.fit(X_boot, y_boot, init=eta, alpha_init=alpha_n, se_estimator=FisherInfoError())
    
    # Return coefficient for ATAC
    return glm_state.beta[atac_idx].item()


def bootstrap_test(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    initial_samples: int,
    key: rdm.PRNGKey,
    p_threshold: float = 0.1,
    atac_idx: int = -1
) -> float:
    """Perform bootstrap test with adaptive sample size
    
    Args:
        X: Design matrix
        y: Response variable
        family: GLM family
        initial_samples: Initial number of bootstrap samples
        key: Random key for reproducibility
        p_threshold: Threshold for increasing bootstrap samples
        atac_idx: Index of ATAC in the design matrix
        
    Returns:
        Bootstrap p-value
    """
    # Get observed coefficient
    glm = GLM(family=family)
    eta, alpha_n = glm.calc_eta_and_dispersion(X, y, jnp.zeros_like(y))
    glm_state = glm.fit(X, y, init=eta, alpha_init=alpha_n, se_estimator=FisherInfoError())
    obs_coef = glm_state.beta[atac_idx].item()
    
    # Initial bootstrap
    n = X.shape[0]
    key, subkey = rdm.split(key)
    indices = rdm.choice(subkey, n, shape=(initial_samples, n), replace=True)
    
    # Use vmap to parallelize bootstrap
    boot_coefs = jax.vmap(lambda idx: bootstrap_regression(X, y, family, idx, atac_idx))(indices)
    
    # Calculate p-value
    p0 = basic_p(obs_coef, boot_coefs)
    
    # Increase bootstrap samples if p-value is small
    if p0 < 0.1 and initial_samples < 500:
        key, subkey = rdm.split(key)
        indices = rdm.choice(subkey, n, shape=(500, n), replace=True)
        boot_coefs = jax.vmap(lambda idx: bootstrap_regression(X, y, family, idx, atac_idx))(indices)
        p0 = basic_p(obs_coef, boot_coefs)
    
    if p0 < 0.05 and initial_samples < 2500:
        key, subkey = rdm.split(key)
        indices = rdm.choice(subkey, n, shape=(2500, n), replace=True)
        boot_coefs = jax.vmap(lambda idx: bootstrap_regression(X, y, family, idx, atac_idx))(indices)
        p0 = basic_p(obs_coef, boot_coefs)
    
    if p0 < 0.01 and initial_samples < 25000:
        key, subkey = rdm.split(key)
        indices = rdm.choice(subkey, n, shape=(25000, n), replace=True)
        boot_coefs = jax.vmap(lambda idx: bootstrap_regression(X, y, family, idx, atac_idx))(indices)
        p0 = basic_p(obs_coef, boot_coefs)
    
    if p0 < 0.001 and initial_samples < 50000:
        key, subkey = rdm.split(key)
        indices = rdm.choice(subkey, n, shape=(50000, n), replace=True)
        boot_coefs = jax.vmap(lambda idx: bootstrap_regression(X, y, family, idx, atac_idx))(indices)
        p0 = basic_p(obs_coef, boot_coefs)
    
    return p0