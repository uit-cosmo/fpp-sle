"""Test module for implementation of variable rate functions generating arrival times."""

import numpy as np
import pytest

from fpp_sle import fpp


def test_inhom_with_valid_rates() -> None:
    """Test sending in an array to the inhom version as the rate function."""
    time_array = np.linspace(0, 10, 100)
    rate = np.sin(np.arange(100))
    rate[rate < 0] = 0

    def rate_func(t):
        return abs(t)

    total_pulses = 10
    arrival_times = fpp.get_arrival_times.from_inhomogeneous_poisson_process(
        rate, time_array, total_pulses
    )
    assert arrival_times.shape == (total_pulses,)
    arrival_times = fpp.get_arrival_times.from_inhomogeneous_poisson_process(
        rate_func, time_array, total_pulses
    )
    assert arrival_times.shape == (total_pulses,)


def test_inhom_too_short_rate_array() -> None:
    """Test the inhom version with a rate array that is too short."""
    total_pulses = 10
    times = np.linspace(0, 10, 100)
    rate = np.array([1, 1])
    with pytest.raises(ValueError):
        fpp.get_arrival_times.from_inhomogeneous_poisson_process(
            rate, times, total_pulses
        )


def test_inhom_negative_rate() -> None:
    """Test the inhom version with a negative rate."""
    total_pulses = 10
    times = np.linspace(0, 10, 100)
    rate = np.array([-1, 0, 1])

    def rate_func(t):
        return -1 + t * 0

    with pytest.raises(ValueError):
        fpp.get_arrival_times.from_inhomogeneous_poisson_process(
            rate, times, total_pulses
        )
        fpp.get_arrival_times.from_inhomogeneous_poisson_process(
            rate_func, times, total_pulses
        )


def test_pass_rate() -> None:
    """Test the `pass_rate` function."""
    time_array = np.linspace(0, 10, 100)
    rate = np.sin(np.arange(100))
    rate[rate < 0] = 0

    def rate_func(t):
        return abs(t)

    total_pulses = 10
    # Must have callable as the first argument
    with pytest.raises(TypeError):
        _ = fpp.get_arrival_times.pass_rate(rate, time_array)
    # Rate as callable
    arrival_times_func = fpp.get_arrival_times.pass_rate(
        fpp.get_arrival_times.from_inhomogeneous_poisson_process, rate_func
    )
    arrival_times = arrival_times_func(time_array, total_pulses)
    assert arrival_times.shape == (total_pulses,)
    # Rate as array
    arrival_times_func = fpp.get_arrival_times.pass_rate(
        fpp.get_arrival_times.from_inhomogeneous_poisson_process, rate
    )
    arrival_times = arrival_times_func(time_array, total_pulses)
    assert arrival_times.shape == (total_pulses,)


if __name__ == "__main__":
    test_inhom_negative_rate()
    test_inhom_with_valid_rates()
    test_inhom_too_short_rate_array()
    test_pass_rate()
