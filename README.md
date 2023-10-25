<h1 align="center">FPP-SLE</h1>
<div align="center">

_**A filtered Poisson process and stochastic logistic equation comparison playground**_

[![PyPI version](https://img.shields.io/pypi/v/fpp-sle)](https://pypi.org/project/fpp-sle/)
[![Python version](https://img.shields.io/pypi/pyversions/fpp-sle)](https://pypi.org/project/fpp-sle/)
[![Licence](https://img.shields.io/badge/license-GPL3-yellow)](https://opensource.org/licenses/GPL-3.0)
[![Tests](https://github.com/uit-cosmo/fpp-sle/workflows/Tests/badge.svg)](https://github.com/uit-cosmo/fpp-sle/actions?workflow=Tests)
[![codecov](https://codecov.io/gh/uit-cosmo/fpp-sle/branch/main/graph/badge.svg?token=F98z2i3T4G)](https://codecov.io/gh/uit-cosmo/fpp-sle)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Install

The package is publised on [PyPI] and installable via `pip`:

```sh
pip install fpp-sle
```

## Usage

See the [examples.py] script for working examples. The main classes and functions this
package provide is

- `VariableRateForcing` (inside `fpp` module)

  This is a class that inherit from the forcing generator class provided by
  [`superposed-pulses`](https://github.com/uit-cosmo/superposed-pulses). The class adds
  a method for setting a custom function that generates arrival times given the time
  axis and a given number of total pulses to generate.

- `get_arrival_times` (inside the `fpp` module)

  This is a module that holds functions that draws arrival times according to some
  non-negative numpy array or callable, that is, the variable rate process.

  - `pass_rate` (inside `get_arrival_times`)

    Used to decorate the functions that draws arrival times from the rate function. This
    is the function you may want to pass in to the `set_arrival_times_function` method
    of the `VariableRateForcing` class. It decorates functions within
    `get_arrival_times` staring with `from_`.

  - `from_` (inside `get_arrival_times`)

    These are generator functions that can take a callable or a numpy array as input,
    and returns arrival times based on the rate function. Currently only one generator
    function is implemented (`from_inhomogeneous_poisson_process`) which draws arrival
    times as if the rate was the underlying rate of a Poisson process.

- `sde`

  This module holds different implementations of stochastic differential equations. See
  the docstring of the individual functions for explanations.

## Contributing

To contribute to the project, clone and install the full development version (uses
[poetry] for dependencies). There is also a `.rtx.toml` file that installs and sets up
an appropriate virtual environment if [rtx](https://github.com/jdx/rtx) is available on
your system (it's really good, check it out!).

```sh
git clone https://github.com/uit-cosmo/fpp-sle.git
cd fpp-sle
# Set up a virtual environment, for example with rtx
rtx i
poetry install
pre-commit install
```

Before committing new changes to a branch you may run command

```sh
nox
```

to run the full test suite. You will need [Poetry], [nox] and [nox-poetry] installed for
this.

[pypi]: https://pypi.org/
[poetry]: https://python-poetry.org
[examples.py]: ./assets/examples.py
[nox]: https://nox.thea.codes/en/stable/
[nox-poetry]: https://nox-poetry.readthedocs.io/
