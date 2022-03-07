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
    n: int,
    x0: float = 0,
    a: Callable[[float, int], float] = lambda x, i: 0 * x * i,
    b: Callable[[float, int], float] = lambda x, i: 1 + 0 * x * i,
    seed: Optional[float] = None,
) -> np.ndarray:
    """Implement the basic Runga-Kutta method for SDEs.

    If a and b depend on some previously defined time vector V(t) (say, a = x(t)
    sqrt(V(t))), use a = lambda x,i: x*np.sqrt(V[i]).

    Make a realization of

    .. math::

        dx(t) = a(x(t)) dt + b(x(t)) dW(t).

    Parameters
    ----------
    dt : float
        Time step.
    n : int
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
    dw = np.random.normal(0, sqdt, n - 1)
    dw2 = 0.5 * (dw**2 - dt) / sqdt

    signal = np.zeros(n)
    signal[0] = x0
    for i in range(n - 1):
        b_instance = b(signal[i], i)
        b_instance2 = b(signal[i] + b_instance * sqdt, i) - b_instance
        signal[i + 1] = (
            signal[i] + a(signal[i], i) * dt + b_instance * dw[i] + b_instance2 * dw2[i]
        )
    return signal


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
    dw = np.random.normal(0, sqdt, n - 1)

    signal = np.zeros(n)
    signal[0] = x0
    for i in range(n - 1):
        signal[i + 1] = signal[i] + theta * (mu - signal[i]) * dt + sigma * dw[i]
    return signal


@nb.jit(nopython=True)
def geometric_brownian_motion(
    dt: float,
    n: int,
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
    n: int
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
    dw = np.random.normal(0, sqdt, n - 1)
    dw2 = 0.5 * (dw**2 - dt) / sqdt

    signal = np.zeros(n)
    signal[0] = x0
    for i in range(n - 1):
        b = sigma * signal[i]
        b2 = sigma * (signal[i] + b * sqdt) - b
        signal[i + 1] = signal[i] + mu * signal[i] * dt + b * dw[i] + b2 * dw2[i]
    return signal


@nb.jit(nopython=True)
def stochastic_logistic_equation(
    dt: float,
    n: int,
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
    n: int
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
    dw = np.random.normal(0, sqdt, n - 1)
    sigma = np.sqrt(2 / (1.0 + gamma))
    signal = np.zeros(n)

    if strong:

        def a(x: np.ndarray) -> np.ndarray:
            return (gamma - np.exp(x)) / (1.0 + gamma)

        # Noise terms
        zeta = np.random.normal(0, sqdt / np.sqrt(3.0), n - 1)
        n1 = 0.75 * sigma * (dw + zeta)
        n2 = 0.5 * sigma * (dw - zeta)

        # Iteration
        signal[0] = np.log(x0)
        for i in range(n - 1):
            h2 = signal[i] + a(signal[i]) * dt
            h3 = 0.75 * signal[i] + 0.25 * (h2 + a(h2) * dt) + n1[i]
            signal[i + 1] = signal[i] / 3.0 + (2.0 / 3.0) * (h3 + a(h3) * dt) + n2[i]
        return np.exp(signal)

    if (not strong) and log:
        signal[0] = np.log(x0)
        for i in range(n - 1):
            signal[i + 1] = (
                signal[i]
                + (gamma - np.exp(signal[i])) * dt / (1.0 + gamma)
                + sigma * dw[i]
            )
        return np.exp(signal)

    signal[0] = x0
    dw2 = 0.5 * (dw**2 - dt) / sqdt
    for i in range(n - 1):
        b = sigma * signal[i]
        b2 = sigma * (signal[i] + b * sqdt) - b
        signal[i + 1] = (
            signal[i]
            + signal[i] * (1.0 - signal[i] / (1.0 + gamma)) * dt
            + b * dw[i]
            + b2 * dw2[i]
        )
    return signal


@nb.jit(nopython=True)
def sde_gexp(
    dt: float,
    n: int,
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
    n: int
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
    dw = np.random.normal(0, sqdt, n - 1)
    signal = np.zeros(n)
    if sqrt:
        signal[0] = np.sqrt(x0)
        for i in range(n - 1):
            signal[i + 1] = (
                signal[i]
                + (0.5 / signal[i]) * (gamma - 0.5 - signal[i] ** 2) * dt
                + dw[i] / np.sqrt(2.0)
            )
        return signal**2
    else:
        signal[0] = x0
        dw2 = 0.5 * (dw**2 - dt) / sqdt
        for i in range(n - 1):
            b = np.sqrt(2 * signal[i])
            b2 = np.sqrt(2 * (signal[i] + b * sqdt)) - b
            signal[i + 1] = (
                signal[i] + (gamma - signal[i]) * dt + b * dw[i] + b2 * dw2[i]
            )
        return signal
