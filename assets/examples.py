"""Example usage of the `VariableRateForcing` class with the `PointModel` class.

You need the development version of this module to run this script.
"""

import cosmoplots
import matplotlib.pyplot as plt
import numpy as np
from model import point_model

from fpp_sle import fpp, sde

__FIG_STD__ = cosmoplots.set_rcparams_dynamo(plt.rcParams)


def rate_function(x):
    high_rate = 100.0
    low_rate = 0.0
    phase_length = 250.0
    arr = low_rate + (high_rate - low_rate) * 0.5 * (
        5 + 10 * np.sin(40 * np.pi * x / phase_length)
    )
    if not isinstance(arr, float):
        arr[arr < 0] = 0
    else:
        arr = arr if arr > 0 else 0
    return arr


def variable_rate_from_array() -> None:
    # Define parameters
    gamma = 1.0
    n = 9999
    dt = 0.01
    total_duration = int(n * dt)

    # Create a rate process. This acts as a varying gamma over the time series.
    rate = sde.ornstein_uhlenbeck(dt=dt, n=n, x0=0.1, theta=1.0, mu=gamma, sigma=1.0)
    rate = abs(rate)

    # We need to create arrival times from this rate process, which are used by the
    # `VariableRateForcing` class. The class knows only about the time axis and the
    # constant gamma value, so we have to provide the rate process corresponding to the
    # constant gamma value, and a way of obtaining arrival times from the rate process.

    # Lets say we want arrival times to simply be the cumulative sum of the rate process:
    # cumsum = fpp.get_arrival_times.from_cumsum
    inhom = fpp.get_arrival_times.from_inhomogeneous_poisson_process
    # We pass in the rate process to this converter function, in addition to a keyword
    # specifying if the rate process should be assumed to have one element per pulse (same
    # shape as the time axis, `same_shape=True`) or if the rate function is much denser
    # (approximating a continuous function compared to the number of pulses).
    # arrival_times_func = fpp.get_arrival_times.pass_rate(cumsum, rate, same_shape=False)
    arrival_times_func = fpp.get_arrival_times.pass_rate(inhom, rate, same_shape=False)
    # The above is equivalent to using a decorator on the `cumsum` function.

    # Now we create the forcing class and specify which arrival time function to use:
    frc = fpp.VariableRateForcing()
    frc.set_arrival_times_function(arrival_times_func)
    sp = point_model.PointModel(gamma=gamma, total_duration=total_duration, dt=dt)
    sp.set_custom_forcing_generator(frc)
    times, signal = sp.make_realization()

    # Plot the results
    fig = plt.figure()
    ax = fig.add_axes(__FIG_STD__)
    arrival_times = sp.get_last_used_forcing().arrival_times
    amplitudes = sp.get_last_used_forcing().amplitudes
    ax.plot(times, rate[: len(times)], label="Rate")
    ax.bar(arrival_times, -1, width=0.1, label="Arrival Times")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rate")
    plt.legend(loc="upper right")
    fig = plt.figure()
    ax = fig.add_axes(__FIG_STD__)
    ax.plot(arrival_times, amplitudes, "o", label="Amplitudes")
    ax.plot(times, signal, label="Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    plt.legend()
    plt.show()


def variable_rate_from_function() -> None:
    # Define parameters
    gamma = 1.0
    n = 9999
    dt = 0.01
    total_duration = int(n * dt)

    # Create a rate process. This acts as a varying gamma over the time series.
    rate = rate_function

    # We need to create arrival times from this rate process, which are used by the
    # `VariableRateForcing` class. The class knows only about the time axis and the
    # constant gamma value, so we have to provide the rate process corresponding to the
    # constant gamma value, and a way of obtaining arrival times from the rate process.

    # Lets say we want arrival times to simply be the cumulative sum of the rate process:
    # cumsum = fpp.get_arrival_times.from_cumsum
    inhom = fpp.get_arrival_times.from_inhomogeneous_poisson_process
    # We pass in the rate process to this converter function, in addition to a keyword
    # specifying if the rate process should be assumed to have one element per pulse (same
    # shape as the time axis, `same_shape=True`) or if the rate function is much denser
    # (approximating a continuous function compared to the number of pulses).
    # arrival_times_func = fpp.get_arrival_times.pass_rate(cumsum, rate, same_shape=False)
    arrival_times_func = fpp.get_arrival_times.pass_rate(inhom, rate, same_shape=False)
    # The above is equivalent to using a decorator on the `cumsum` function.

    # Now we create the forcing class and specify which arrival time function to use:
    frc = fpp.VariableRateForcing()
    frc.set_arrival_times_function(arrival_times_func)
    sp = point_model.PointModel(gamma=gamma, total_duration=total_duration, dt=dt)
    sp.set_custom_forcing_generator(frc)
    times, signal = sp.make_realization()

    # Plot the results
    fig = plt.figure()
    ax = fig.add_axes(__FIG_STD__)
    arrival_times = sp.get_last_used_forcing().arrival_times
    amplitudes = sp.get_last_used_forcing().amplitudes
    ax.plot(times, rate(times) / np.max(rate(times)) * max(amplitudes), label="Rate")
    ax.bar(arrival_times, -1, width=0.1, label="Arrival Times")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rate")
    plt.legend(loc="upper right")
    fig = plt.figure()
    ax = fig.add_axes(__FIG_STD__)
    ax.plot(arrival_times, amplitudes, "o", label="Amplitudes")
    ax.plot(times, signal, label="Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    variable_rate_from_function()
    variable_rate_from_array()
