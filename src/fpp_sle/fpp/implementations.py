"""Module implementing superposed pulses."""

from typing import Callable

import numpy as np
from model import forcing, point_model

from fpp_sle import sde
from fpp_sle.fpp import get_arrival_times


class VariableRateForcing(forcing.ForcingGenerator):
    """Class that implements a variable rate forcing.

    By default, amplitudes are drawn from an exponential distribution and duration times
    are set to 1, but this can be overridden by the user by calling the methods
    `set_amplitude_distribution` and `set_duration_distribution`.
    """

    def __init__(self):
        self._amplitude_distribution = None
        self._duration_distribution = None

    def get_forcing(self, times: np.ndarray, gamma: float) -> forcing.Forcing:
        """Generate the forcing.

        Arrival times are generated using the function `get_arrival_times`. See the module
        `fpp_sle.fpp.get_arrival_times` for example implementations.

        Parameters
        ----------
        times : np.ndarray
            The times at which the forcing is to be generated.
        gamma : float
            Intermittency parameter, long term mean of the rate process.

        Returns
        -------
        forcing.Forcing
            The generated forcing.
        """
        arrival_times = self._get_arrival_times(times, gamma)
        total_pulses = int(max(times) * gamma)
        # FIXME: The Forcing class checks if the provided `total_pulses` is equal to the
        # length of the `arrival_times` array.
        amplitudes = self._get_amplitudes(total_pulses)
        durations = self._get_durations(total_pulses)
        return forcing.Forcing(total_pulses, arrival_times, amplitudes, durations)

    def set_arrival_times_function(
        self, arrival_times_function: Callable[[np.ndarray, float], np.ndarray]
    ):
        self._arrival_times_function = arrival_times_function

    def set_amplitude_distribution(
        self,
        amplitude_distribution_function: Callable[[int], np.ndarray],
    ):
        self._amplitude_distribution = amplitude_distribution_function

    def set_duration_distribution(
        self, duration_distribution_function: Callable[[int], np.ndarray]
    ):
        self._duration_distribution = duration_distribution_function

    def _get_arrival_times(self, times, gamma) -> np.ndarray:
        """Generate the arrival times.

        Parameters
        ----------
        times : np.ndarray
            The times at which the forcing is to be generated.
        gamma : float
            Intermittency parameter, long term mean of the rate process.

        Returns
        -------
        np.ndarray
            The arrival times.

        Raises
        ------
        NotImplementedError
            If the arrival times function has not been set.
        """
        if self._arrival_times_function is not None:
            return self._arrival_times_function(times, gamma)
        raise NotImplementedError(
            "No arrival times function has been set. "
            "Use `set_arrival_times_function` to set one."
        )

    def _get_amplitudes(self, total_pulses) -> np.ndarray:
        if self._amplitude_distribution is not None:
            return self._amplitude_distribution(total_pulses)
        return np.random.default_rng().exponential(scale=np.ones(total_pulses))

    def _get_durations(self, total_pulses) -> np.ndarray:
        if self._duration_distribution is not None:
            return self._duration_distribution(total_pulses)
        return np.ones(total_pulses)


if __name__ == "__main__":
    gamma = 1.0
    rate = sde.ornstein_uhlenbeck(dt=1, n=10, x0=0.1, theta=1.0, mu=gamma, sigma=1.0)
    arrival_times_func = get_arrival_times.pass_rate(
        get_arrival_times.as_cumsum, rate, one_to_one=True
    )
    frc = VariableRateForcing()
    frc.set_arrival_times_function(arrival_times_func)
    sp = point_model.PointModel(gamma=gamma, total_duration=10000, dt=0.01)
    sp.set_custom_forcing_generator(frc)
    times, signal = sp.make_realization()
