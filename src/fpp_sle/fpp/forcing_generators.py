"""Implementation of superposed pulses variable rate forcing class."""

from typing import Callable

import numpy as np
from superposedpulses import forcing


class VariableRateForcing(forcing.ForcingGenerator):
    """Class that implements a variable rate forcing.

    By default, amplitudes are drawn from an exponential distribution and duration times
    are set to 1, but this can be overridden by the user by calling the methods
    `set_amplitude_distribution` and `set_duration_distribution`.
    """

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
        total_pulses = int(max(times) * gamma)
        arrival_times = self._get_arrival_times(times, total_pulses)
        amplitudes = self._get_amplitudes(total_pulses)
        durations = self._get_durations(total_pulses)
        return forcing.Forcing(total_pulses, arrival_times, amplitudes, durations)

    def set_arrival_times_function(
        self, f: Callable[[np.ndarray, int], np.ndarray]
    ) -> None:
        """Set the arrival times function.

        Parameters
        ----------
        f : Callable[[np.ndarray, int], np.ndarray]
            The arrival times function.
        """
        self._arrival_times_function = f

    def set_amplitude_distribution(self, f: Callable[[int], np.ndarray]) -> None:
        """Set the amplitude distribution function.

        Parameters
        ----------
        f : Callable[[int], np.ndarray]
            The amplitude distribution function.
        """
        self._amplitude_distribution = f

    def set_duration_distribution(self, f: Callable[[int], np.ndarray]) -> None:
        """Set the duration distribution function.

        Parameters
        ----------
        f : Callable[[int], np.ndarray]
            The duration distribution function.
        """
        self._duration_distribution = f

    def _get_arrival_times(self, times: np.ndarray, total_pulses: int) -> np.ndarray:
        """Generate the arrival times.

        Parameters
        ----------
        times : np.ndarray
            The times at which the forcing is to be generated.
        total_pulses : int
            The total number of pulses.

        Returns
        -------
        np.ndarray
            The arrival times.

        Raises
        ------
        AttributeError
            If the arrival times function has not been set.
        """
        if hasattr(self, "_arrival_times_function"):
            return self._arrival_times_function(times, total_pulses)
        raise AttributeError(
            "'VariableRateForcing' object has no attribute '_arrival_times_function'. "
            "Did you mean: 'set_arrival_times_function'?"
        )

    def _get_amplitudes(self, total_pulses: int) -> np.ndarray:
        if hasattr(self, "_amplitude_distribution"):
            return self._amplitude_distribution(total_pulses)
        return np.random.default_rng().exponential(scale=np.ones(total_pulses))

    def _get_durations(self, total_pulses: int) -> np.ndarray:
        if hasattr(self, "_duration_distribution"):
            return self._duration_distribution(total_pulses)
        return np.ones(total_pulses)
