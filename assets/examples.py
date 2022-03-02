"""Example usage of the `VariableRateForcing` class with the `PointModel` class.

You need the development version of this module to run this script.
"""

import cosmoplots
import matplotlib.pyplot as plt
from model import point_model

from fpp_sle import fpp, sde

__FIG_STD__ = cosmoplots.set_rcparams_dynamo(plt.rcParams)


def variable_rate_forcing() -> None:
    # Define parameters
    gamma = 1.0
    n = 9999
    dt = 0.01
    total_duration = int(n * dt)

    # Create a rate process. This acts as a varying gamma over the time series.
    rate = sde.ornstein_uhlenbeck(dt=dt, n=n, x0=0.1, theta=1.0, mu=gamma, sigma=1.0)
    # We need to create arrival times from this rate process, which are used by the
    # `VariableRateForcing` class. The class knows only about the time axis and the
    # constant gamma value, so we have to provide the rate process corresponding to the
    # constant gamma value, and a way of obtaining arrival times from the rate process.

    # Lets say we want arrival times to simply be the cumulative sum of the rate process:
    cumsum = fpp.get_arrival_times.as_cumsum
    # We pass in the rate process to this converter function, in addition to a keyword
    # specifying if the rate process should be assumed to have one element per pulse (same
    # shape as the time axis, `same_shape=True`) or if the rate function is much denser
    # (approximating a continuous function compared to the number of pulses).
    arrival_times_func = fpp.get_arrival_times.pass_rate(cumsum, rate, same_shape=False)
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
    ax.plot(arrival_times, amplitudes, "o", label="Amplitudes")
    ax.plot(times, signal, label="Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    variable_rate_forcing()
