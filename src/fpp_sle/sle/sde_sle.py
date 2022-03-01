"""Implementation of the stochastic logistic equation."""

from fpp_sle import sde


def main() -> None:
    """Main function."""
    _ = sde.ornstein_uhlenbeck(0.0, 0, 0.0, 0.0)


if __name__ == "__main__":
    main()
