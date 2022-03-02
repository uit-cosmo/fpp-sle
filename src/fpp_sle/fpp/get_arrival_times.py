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

from typing import Callable, Tuple

import numpy as np


def check_types(func) -> Callable[[np.ndarray, bool, np.ndarray, int], np.ndarray]:
    """Check that the types of the arguments and return value are correct.

    Parameters
    ----------
    func: Callable
        The function to decorate.

    Returns
    -------
    Callable[[np.ndarray, bool, np.ndarray, int], np.ndarray]
        The decorated function.
    """
    # fmt: off
    def check_types_wrapper(*args, **kwargs) -> np.ndarray:
        if not isinstance(args[0], np.ndarray):
            raise TypeError(f"First argument must be a numpy array (rate), found {type(args[0])}.")
        if not isinstance(args[1], bool):
            raise TypeError(f"Second argument must be a bool (same_shape), found {type(args[1])}.")
        if not isinstance(args[2], np.ndarray):
            raise TypeError(f"Third argument must be a numpy array (times), found {type(args[2])}.")
        if not isinstance(args[3], int):
            raise TypeError(f"Fourth argument must be an int (total_pulses), found {type(args[3])}.")
        return func(*args, **kwargs)
    # fmt: on
    return check_types_wrapper


def pass_rate(func, rate, same_shape=True) -> Callable[[np.ndarray, int], np.ndarray]:
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


@check_types
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


@check_types
def as_cox_process(
    rate: np.ndarray, same_shape: bool, times: np.ndarray, mu: float
) -> Tuple[np.ndarray, float]:
    raise NotImplementedError


@check_types
def as_inhomogenous_poisson_process(
    rate: np.ndarray, same_shape: bool, times: np.ndarray, mu: float
) -> Tuple[np.ndarray, float]:
    raise NotImplementedError
