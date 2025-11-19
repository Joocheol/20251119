"""Monte Carlo option pricing with customizable payoff expressions.

Usage example::

    python monte_carlo_option.py --s0 100 --r 0.03 --sigma 0.2 --t 1 --paths 10000 \
        --payoff "max(ST - 110, 0)"

The payoff expression is evaluated for each simulated terminal price ``ST``. The
expression can also use ``S0``, ``K`` (if provided), ``r``, ``sigma``, ``T``, and
``np`` (NumPy) for convenience.
"""
from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo option pricer")
    parser.add_argument("--s0", type=float, required=True, help="Initial underlying price")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate (annualized)")
    parser.add_argument("--sigma", type=float, required=True, help="Volatility (annualized)")
    parser.add_argument("--t", type=float, required=True, help="Time to maturity in years")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of time steps in each simulated path (must be >= 1)",
    )
    parser.add_argument(
        "--paths", type=int, default=10000, help="Number of Monte Carlo paths to simulate"
    )
    parser.add_argument(
        "--payoff",
        type=str,
        required=True,
        help=(
            "Payoff expression evaluated for each terminal price. Use ST for the "
            "terminal price, e.g. 'max(ST - 100, 0)'."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=None,
        help="Strike price (optional, but provided for convenience in payoff expressions)",
    )
    return parser.parse_args()


def simulate_gbm_terminal_prices(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
    paths: int,
    seed: int | None = None,
) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if seed is not None:
        np.random.seed(seed)
    dt = t / steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    increments = drift + diffusion * np.random.standard_normal(size=(paths, steps))
    log_s = np.log(s0) + np.cumsum(increments, axis=1)
    terminal_prices = np.exp(log_s[:, -1]) if steps > 0 else np.full(paths, s0)
    return terminal_prices


def evaluate_payoff(
    payoff_expr: str,
    terminal_prices: np.ndarray,
    context: Dict[str, float],
) -> np.ndarray:
    """Evaluate the user-specified payoff expression for each terminal price."""
    allowed_globals = {"__builtins__": {}}
    allowed_locals = {
        "np": np,
        "numpy": np,
        "max": max,
        "min": min,
        "abs": abs,
    }
    allowed_locals.update(context)
    code = compile(payoff_expr, "<payoff>", "eval")
    payoffs = np.empty_like(terminal_prices)
    for idx, st in enumerate(terminal_prices):
        allowed_locals["ST"] = st
        payoffs[idx] = float(eval(code, allowed_globals, allowed_locals))
    return payoffs


def price_option(args: argparse.Namespace) -> Dict[str, float]:
    terminal_prices = simulate_gbm_terminal_prices(
        s0=args.s0,
        r=args.r,
        sigma=args.sigma,
        t=args.t,
        steps=args.steps,
        paths=args.paths,
        seed=args.seed,
    )

    payoff_context = {
        "S0": args.s0,
        "K": args.k if args.k is not None else 0.0,
        "r": args.r,
        "sigma": args.sigma,
        "T": args.t,
    }

    payoffs = evaluate_payoff(args.payoff, terminal_prices, payoff_context)
    discounted = np.exp(-args.r * args.t) * payoffs
    mean_price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(args.paths))
    return {"price": mean_price, "stderr": stderr}


def price_option_from_params(
    *,
    s0: float,
    r: float,
    sigma: float,
    t: float,
    payoff: str,
    steps: int = 1,
    paths: int = 10000,
    seed: int | None = None,
    k: float | None = None,
) -> Dict[str, float]:
    """Helper wrapper so other modules (like the web app) can reuse pricing logic."""

    args = argparse.Namespace(
        s0=s0,
        r=r,
        sigma=sigma,
        t=t,
        steps=steps,
        paths=paths,
        payoff=payoff,
        seed=seed,
        k=k,
    )
    return price_option(args)


def main() -> None:
    args = parse_arguments()
    result = price_option(args)
    print(f"Monte Carlo price: {result['price']:.6f}")
    print(f"Standard error : {result['stderr']:.6f}")


if __name__ == "__main__":
    main()
