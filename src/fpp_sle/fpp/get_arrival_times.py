"""Create arrival times from rate processes.

The rate process is understood as a varying intermittency parameter used within the FPP
framework.

All functions in this module (starting with `from_`) should be decorated with `pass_rate`
when used with the `VariableRateForcing` class. They also take in the same set of
parameters:
    rate: np.ndarray
        The rate process to convert.
    same_shape: bool
        If True, the rate process is assumed to be the same length as the time array.
        Defaults to True.
    times: np.ndarray
        The time axis.
    total_pulses: int
        The total number of pulses.
"""

from typing import Any, Callable, Tuple, Union

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
        if not isinstance(args[0], np.ndarray) and not callable(args[0]):
            raise TypeError(f"First argument must be a numpy array or a callable (rate), found {type(args[0])}.")
        if not isinstance(args[1], np.ndarray):
            raise TypeError(f"Second argument must be a numpy array (times), found {type(args[1])}.")
        if not isinstance(args[2], int):
            raise TypeError(f"Third argument must be an int (total_pulses), found {type(args[2])}.")
        return func(*args, **kwargs)
    # fmt: on
    return check_types_wrapper


def pass_rate(func, rate, **kwargs: Any) -> Callable[[np.ndarray, int], np.ndarray]:
    """Decorator function to pass the rate process to a function.

    This will also specify if the rate process has the same length as the time array.

    Parameters
    ----------
    func: function
        The function to decorate.
    rate: np.ndarray
        The rate process to pass on to the function.
    **kwargs: Any
        Additional keyword arguments to pass to the function.

    Returns
    -------
    function
        The decorated function.
    """

    def inner(times: np.ndarray, total_pulses: int) -> np.ndarray:
        return func(rate, times, total_pulses, **kwargs)

    return inner


@check_types
def from_cumsum(
    rate: Union[Callable[..., Union[float, np.ndarray]], np.ndarray],
    times: np.ndarray,
    total_pulses: int,
    same_shape: bool = True,
    **kwargs: Any,
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
    rate: Union[Callable, np.ndarray]
        The rate process to convert.
    times: np.ndarray
        The time axis.
    total_pulses: int
        The total number of pulses to generate.
    same_shape: bool
        If True, the rate process is assumed to be the same length as the time array.
        Defaults to True.
    **kwargs: Any
        Additional keyword arguments to pass to the rate process.

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
        "WARNING: Function `fpp.get_arrival_times.from_cumsum` is not correct. "
        + "Just used as a placeholder."
    )
    if isinstance(rate, np.ndarray):
        rate_realisaition = rate
    else:
        rate_realisaition = np.array(rate(times, **kwargs))
    if not same_shape:
        ratio = max(int(len(rate_realisaition) / total_pulses), 1)
        rate_realisaition = rate_realisaition[::ratio][:total_pulses]
    if len(rate_realisaition) != total_pulses:
        raise ValueError(
            f"Rate process is shorter than time array. Found {len(rate_realisaition) = } < {total_pulses = }."
        )
    return np.cumsum(rate_realisaition) / rate_realisaition.sum() * times[-1]


@check_types
def from_cox_process(
    rate: np.ndarray, same_shape: bool, times: np.ndarray, mu: float
) -> Tuple[np.ndarray, float]:
    raise NotImplementedError


@check_types
def from_inhomogeneous_poisson_process(
    rate: Union[Callable[..., Union[float, np.ndarray]], np.ndarray],
    times: np.ndarray,
    total_pulses: int,
    same_shape: bool = False,
    **kwargs,
) -> np.ndarray:
    if isinstance(rate, np.ndarray):
        rate_realisaition = rate
        if not same_shape:
            ratio = max(int(len(rate_realisaition) / total_pulses), 1)
            rate_realisaition = rate_realisaition[::ratio]
        return rate_realisaition[:total_pulses]
    else:
        arrival_times = np.array([])
        while len(arrival_times) < total_pulses:
            delta = times[1] - times[0]
            t = np.arange(delta, times[-1], delta)
            avg_rate = (rate(t, **kwargs) + rate(t + delta, **kwargs)) / 2.0
            avg_prob = 1 - np.exp(-avg_rate * delta / 1000.0)
            rand_throws = np.random.uniform(size=t.shape[0])
            arrival_times = np.concatenate((arrival_times, t[rand_throws < avg_prob]))
        arrival_times.sort()
        arrival_times = arrival_times[:total_pulses]

        return arrival_times
