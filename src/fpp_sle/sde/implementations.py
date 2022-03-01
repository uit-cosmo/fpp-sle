"""Module implementing different stochastic differential equations as numba jit functions.

Originally implemented in the abandoned project `uit_scripts`, in the `runge_kutta_SDE`
module. Re-formatted with type hints and re-implemented in `fpp_sle` module.
"""

from typing import Callable, Optional

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def general_sde(
    dt: float,
    N: int,
    x0: float = 0,
    a: Callable[[float, int], float] = lambda x, i: 0 * x * i,
    b: Callable[[float, int], float] = lambda x, i: 1 + 0 * x * i,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Implementation of the basic Runga-Kutta method for SDEs.

    If a and b depend on some previously defined time vector V(t) (say, a = x(t)
    sqrt(V(t))), use a = lambda x,i: x*np.sqrt(V[i]).

    Make a realization of

    .. math::

        dx(t) = a(x(t)) dt + b(x(t)) dW(t).

    Parameters
    ----------
    dt : float
        Time step.
    N : int
        Number of time steps/iterations.
    x0 : float
        Initial value.
    a : Callable[[float, int], float]
        Function that returns the value of the drift term.
    b : Callable[[float, int], float]
        Function that returns the value of the diffusion term.
    seed : float
        Seed for the random number generator.

    Returns
    -------
    np.array
        Array of size N with the realization of the SDE.

    Notes
    -----
    For background on the Runge-Kutta method, see
    An introduction to numerical methods for stochastic differential equations, E. Platen,
    Acta Numerica 8 (1999)
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0, sqdt, N - 1)
    dW2 = 0.5 * (dW**2 - dt) / sqdt

    X = np.zeros(N)
    X[0] = x0
    for i in range(N - 1):
        B = b(X[i], i)
        B2 = b(X[i] + B * sqdt, i) - B
        X[i + 1] = X[i] + a(X[i], i) * dt + B * dW[i] + B2 * dW2[i]
    return X


@nb.jit(nopython=True)
def ornstein_uhlenbeck(
    dt: float,
    n: int,
    x0: float = 0.0,
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Fast calculation of the Ornstein-Uhlenbeck process.

    Parameters
    ----------
    dt: float
        Time step.
    n: int
        Number of time steps/iterations.
    x0: float
        Initial state.
    theta: float
        Speed of reversion.
    mu: float
        Long term mean.
    sigma: float
        Volatility.
    seed: float
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        The Ornstein-Uhlenbeck process.

    Notes
    -----
    This function is a numba jit function. It solves an SDE numerically to create an
    Ornstein-Uhlenbeck process with parameters theta, mu and sigma:

    .. math::

        dx_t = theta(mu-x_t)dt + sigma dW_t

    The SDE method is Euler-Maruyama.
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0, sqdt, n - 1)

    X = np.zeros(n)
    X[0] = x0
    for i in range(n - 1):
        X[i + 1] = X[i] + theta * (mu - X[i]) * dt + sigma * dW[i]
    return X


@nb.jit(nopython=True)
def geometric_brownian_motion(
    dt: float,
    N: int,
    x0: float = 0.0,
    mu: float = 1.0,
    sigma: float = 1.0,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Fast calculation of the geometric Brownian motion.

    Uses parameters mu and sigma:

    .. math::

        dx_t = mu x_t dt + sigma x_t dW_t

    Parameters
    ----------
    dt: float
        Time step.
    N: int
        Number of time steps/iterations.
    x0: float
        Initial state.
    mu: float
        Long term mean.
    sigma: float
        Volatility.
    seed: float
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        The geometric Brownian motion.
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0, sqdt, N - 1)
    dW2 = 0.5 * (dW**2 - dt) / sqdt

    X = np.zeros(N)
    X[0] = x0
    for i in range(N - 1):
        B = sigma * X[i]
        B2 = sigma * (X[i] + B * sqdt) - B
        X[i + 1] = X[i] + mu * X[i] * dt + B * dW[i] + B2 * dW2[i]
    return X


@nb.jit(nopython=True)
def stochastic_logistic_equation(
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


@nb.jit(nopython=True)
def SDE_GEXP(
    dt: float,
    N: int,
    x0: float = 2.0,
    gamma: float = 2.0,
    sqrt=True,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Fast calculation of a gamma distributed process.

    Uses shape parameter gamma, scale parameter 1. The generated process has exponential
    autocorrelation function:

    .. math::

        dx_t = (gamma-x_t) dt + sqrt(2 x_t) dW_t

    OBS: This is numerically stable only for gamma > 1.5 or so (for dt = 1e-3, the
    stability threshold may depend on dt).

    Parameters
    ----------
    dt: float
        Time steps.
    N: int
        Number of time steps/iterations.
    x0: float
        Initial state.
    gamma: float
        Shape parameter.
    sqrt: bool
        If True, estimate sqrt(x_t) and then square it.
    seed: float
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        The gamma distributed process.
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0, sqdt, N - 1)
    X = np.zeros(N)
    if sqrt:
        X[0] = np.sqrt(x0)
        for i in range(N - 1):
            X[i + 1] = (
                X[i]
                + (0.5 / X[i]) * (gamma - 0.5 - X[i] ** 2) * dt
                + dW[i] / np.sqrt(2.0)
            )
        return X**2
    else:
        X[0] = x0
        dW2 = 0.5 * (dW**2 - dt) / sqdt
        for i in range(N - 1):
            B = np.sqrt(2 * X[i])
            B2 = np.sqrt(2 * (X[i] + B * sqdt)) - B
            X[i + 1] = X[i] + (gamma - X[i]) * dt + B * dW[i] + B2 * dW2[i]
        return X
