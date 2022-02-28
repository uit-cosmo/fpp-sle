"""Implementation of the stochastic logistic equation."""


from typing import Optional

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def SDE_SLE(
    dt: float,
    N: int,
    x0: float = 1.0,
    gamma: float = 1.0,
    log: bool = False,
    strong: bool = True,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Fast calculation of the stochastic logistic equation.

    This implementation uses a parameter gamma that acts as an intermittency parameter for
    the resulting stochastic process.

    .. math::

        dx_t = x_t(1-x_t/(1.+gamma)) dt + sqrt(2/(1+gamma)) x_t dW_t

    Parameters
    ----------
    dt: float
        Time step
    N: int
        Number of iterations
    x0: float
        Initial state
    gamma: float
        Intermittency parameter
    log: bool
        If True, estimate log(x_t) and then take the exponential.
    strong: bool
        If True, estimate the logarithm using the strong order 1.5, weak order 3 algorithm
        of A. Rossler DOI:10.1137/09076636X
    seed: float
        Seed for random state

    Returns
    -------
    np.ndarray
        Array of the stochastic logistic equation

    Notes
    -----
    This function was originally implemented in uit-cosmo/uit_scripts. This implementation
    is based on the following paper: Rossler DOI:10.1137/09076636X
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0, sqdt, N - 1)
    sigma = np.sqrt(2 / (1.0 + gamma))
    X = np.zeros(N)

    if strong:

        def a(x: np.ndarray) -> np.ndarray:
            return (gamma - np.exp(x)) / (1.0 + gamma)

        # Noise terms
        zeta = np.random.normal(0, sqdt / np.sqrt(3.0), N - 1)
        N1 = 0.75 * sigma * (dW + zeta)
        N2 = 0.5 * sigma * (dW - zeta)

        # Iteration
        X[0] = np.log(x0)
        for i in range(N - 1):
            H2 = X[i] + a(X[i]) * dt
            H3 = 0.75 * X[i] + 0.25 * (H2 + a(H2) * dt) + N1[i]
            X[i + 1] = X[i] / 3.0 + (2.0 / 3.0) * (H3 + a(H3) * dt) + N2[i]
        return np.exp(X)

    if (not strong) and log:
        X[0] = np.log(x0)
        for i in range(N - 1):
            X[i + 1] = (
                X[i] + (gamma - np.exp(X[i])) * dt / (1.0 + gamma) + sigma * dW[i]
            )
        return np.exp(X)

    X[0] = x0
    dW2 = 0.5 * (dW**2 - dt) / sqdt
    for i in range(N - 1):
        B = sigma * X[i]
        B2 = sigma * (X[i] + B * sqdt) - B
        X[i + 1] = (
            X[i] + X[i] * (1.0 - X[i] / (1.0 + gamma)) * dt + B * dW[i] + B2 * dW2[i]
        )
    return X
