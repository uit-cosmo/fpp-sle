"""Create arrival times from rate processes.

The rate process is understood as a varying intermittency parameter used within the FPP
framework.

All functions in this module (starting with `from_`) should be decorated with `pass_rate`
when used with the `VariableRateForcing` class. They also take in the same set of
parameters:
    rate: np.ndarray
        The rate process to convert.
    times: np.ndarray
        The time axis.
    total_pulses: int
        The total number of pulses.
"""

from typing import Any, Callable, Union

import numpy as np


def check_types(
    func,
) -> Callable[
    [Union[Callable[..., Union[float, np.ndarray]], np.ndarray], np.ndarray, int],
    np.ndarray,
]:
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

    def check_types_wrapper(*args, **kwargs) -> np.ndarray:
        if not isinstance(args[0], np.ndarray) and not callable(args[0]):
            txt = "First argument must be a numpy array or a callable (rate), found"
            raise TypeError(f"{txt} {type(args[0])}.")
        elif isinstance(args[0], np.ndarray) and any(args[0] < 0):
            txt = f"The rate process must be non-negative, found {min(args[0])}."
            raise ValueError(txt)
        if not isinstance(args[1], np.ndarray):
            txt = (
                f"Second argument must be a numpy array (times), found {type(args[1])}."
            )
            raise TypeError(txt)
        if not isinstance(args[2], int):
            txt = (
                f"Third argument must be an int (total_pulses), found {type(args[2])}."
            )
            raise TypeError(txt)
        return func(*args, **kwargs)

    return check_types_wrapper


def pass_rate(func, rate, **kwargs: Any) -> Callable[[np.ndarray, int], np.ndarray]:
    """Decorate a function by passing the in a rate process.

    This will also specify if the rate process has the same length as the time array.

    Parameters
    ----------
    func: function
        The function to decorate.
    rate: np.ndarray
        The rate process to pass on to the function.
    kwargs: Any
        Additional keyword arguments to pass to the function.

    Returns
    -------
    function
        The decorated function.

    Raises
    ------
    TypeError
        If the first argument is not a function (should be a function to get arrival
        times).
    """
    if not callable(func):
        raise TypeError(f"The first argument must be a function, found {type(func)}.")

    def inner(times: np.ndarray, total_pulses: int) -> np.ndarray:
        """Pass on arguments to the decorated function.

        This is the function that is sent to the `VariableRateForcing` class.

        Parameters
        ----------
        times: np.ndarray
            The time axis.
        total_pulses: int
            The total number of pulses.

        Returns
        -------
        np.ndarray
            The arrival times.
        """
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
    kwargs: Any
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
        + "Just used as a placeholder. Use `from_inhomogeneous_poisson_process` instead."
    )
    if isinstance(rate, np.ndarray):
        rate_realization = rate
    else:
        rate_realization = np.array(rate(times, **kwargs))
    if not same_shape:
        ratio = max(len(rate_realization) // total_pulses, 1)
        rate_realization = rate_realization[::ratio][:total_pulses]
    if len(rate_realization) != total_pulses:
        raise ValueError(
            "Rate process is shorter than time array. "
            + f"Found {len(rate_realization) = } < {total_pulses = }."
        )
    return np.cumsum(rate_realization) / rate_realization.sum() * times[-1]


@check_types
def from_inhomogeneous_poisson_process(
    rate: Union[Callable[..., Union[float, np.ndarray]], np.ndarray],
    times: np.ndarray,
    total_pulses: int,
    **kwargs: Any,
) -> np.ndarray:
    """Convert a rate process to arrival times as an inhomogeneous Poisson process.

    The rate process is understood as a varying intermittency parameter used within the
    FPP framework. If enough arrival times are not obtained from the rate process during
    the first iteration, the algorithm will repeat the process and append the new arrival
    times until there are more arrival times than total pulses. A `total_pulses` number of
    arrival times are then uniformly sampled from the resulting process.

    Parameters
    ----------
    rate: Union[Callable[..., Union[float, np.ndarray]], np.ndarray],
        The rate process to convert. Can be a precomputed numpy array, or a callable. If
        it is a callable, it must return a float or numpy array (depending on the input).
        Any keyword arguments passed to `from_inhomogeneous_poisson_process` will be
        passed on to the rate process callable.
    times: np.ndarray
        The time axis.
    total_pulses: int
        The total number of pulses to generate.
    kwargs: Any
        Additional keyword arguments to pass to the rate process.

    Returns
    -------
    np.ndarray
        The arrival times.

    Raises
    ------
    ValueError
        If the rate process is shorter than the time array.

    Notes
    -----
    The algorithm is based on the following discussion:
    https://stackoverflow.com/a/32713988
    """
    if isinstance(rate, np.ndarray):
        if len(rate) < len(times):
            raise ValueError(
                f"Rate process is shorter than time array. Found {len(rate) = } < {len(times) = }."
            )
        elif len(rate) != len(times):
            # If the rate process is longer than the time axis, we split the rate process
            # in as large blocks as possible while making sure we get the same number of
            # blocks as we have time steps, then we take the average of the blocks.
            new_length = len(rate) // len(times)
            nearest_multiple = int(len(times) * new_length)
            reshaped = rate[:nearest_multiple].reshape((len(times), new_length))
            rate = np.sum(reshaped, axis=1)
    arrival_times = np.array([])
    while len(arrival_times) < total_pulses:
        delta = times[1] - times[0]
        if isinstance(rate, np.ndarray):
            avg_rate = rate[:]
        else:
            avg_rate = (rate(times, **kwargs) + rate(times + delta, **kwargs)) / 2.0
        avg_prob = 1 - np.exp(-avg_rate * delta)
        rand_throws = np.random.uniform(size=times.shape[0])
        arrival_times = np.concatenate((arrival_times, times[rand_throws < avg_prob]))
    arrival_times = np.random.choice(arrival_times, size=total_pulses)
    arrival_times.sort()
    return arrival_times
