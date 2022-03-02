"""Create arrival times from rate processes.

The rate process is understood as a varying intermittency parameter used within the FPP
framework.

All functions in this module (starting with `as_`) should be decorated with `pass_rate` when used with the
`VariableRateForcing` class. They also take in the same set of parameters:
    rate: np.ndarray
        The rate process to convert.
    same_shape: bool
        If True, the rate process is assumed to be the same length as the time array.
        Defaults to True.
    times: np.ndarray
        The time axis.
    mu: float
        The mean of the rate process.
"""

from typing import Tuple

import numpy as np


def pass_rate(func, rate, same_shape=True):
    """Decorator function to pass the rate process to a function.

    This will also specify if the rate process has the same length as the time array.

    Parameters
    ----------
    func: function
        The function to decorate.
    rate: np.ndarray
        The rate process to pass on to the function.
    same_shape: bool
        If True, the rate process is assumed to be the same length as the time array.
        Defaults to True.

    Returns
    -------
    function
        The decorated function.
    """

    def inner(*args) -> np.ndarray:
        return func(rate, same_shape, *args)

    return inner


def as_cumsum(
    rate: np.ndarray, same_shape: bool, times: np.ndarray, total_pulses: int
) -> np.ndarray:
    """Convert a rate process to arrival times.

    Arrival times should be in ascending order, ending at `times[-1]`. We here assume
    `gamma` to be varying over the time interval, and that a sense of randomness is
    already present in the gamma. Therefore, we just compute the cumulative sum and scale
    the arrival times accordingly.

    If the rate process is long, we choose every `K`th element as the arrival time, such
    that `len(rate)==len(times)`.

    Parameters
    ----------
    rate: np.ndarray
        The rate process to convert.
    same_shape: bool
        If True, the rate process is assumed to be the same length as the time array.
        Defaults to True.
    times: np.ndarray
        The time axis.
    total_pulses: int
        The total number of pulses to generate.

    Returns
    -------
    np.ndarray
        The cumulative-cumulative process.

    Raises
    ------
    ValueError
        If the rate process is shorter than the time array.
    """
    # FIXME: High rate mean few arrivals, while the opposite should be the case.
    print(
        "WARNING: Function `fpp.get_arrival_times.as_cumsum` is not correct. "
        + "Just used as a placeholder."
    )
    if not same_shape:
        ratio = max(int(len(rate) / total_pulses), 1)
        rate = rate[::ratio][:total_pulses]
    if len(rate) != total_pulses:
        raise ValueError(
            f"Rate process is shorter than time array. Found {len(rate) = } < {total_pulses = }."
        )
    return np.cumsum(rate) / rate.sum() * times[-1]


def as_cox_process(
    rate: np.ndarray, same_shape: bool, times: np.ndarray, mu: float
) -> Tuple[np.ndarray, float]:
    raise NotImplementedError


def as_inhomogenous_poisson_process(
    rate: np.ndarray, same_shape: bool, times: np.ndarray, mu: float
) -> Tuple[np.ndarray, float]:
    raise NotImplementedError
