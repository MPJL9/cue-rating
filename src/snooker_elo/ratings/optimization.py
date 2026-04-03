"""MLE parameter optimization for rating systems.

Finds optimal parameters by maximizing the log-likelihood of observed
match outcomes. Each frame is treated as a Bernoulli trial with probability
predicted by the rating system.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from snooker_elo.ratings.elo import EloRating
from snooker_elo.ratings.glicko2 import Glicko2Rating


def _compute_log_likelihood(
    matches: pd.DataFrame,
    make_system: Callable[[], EloRating | Glicko2Rating],
    eval_start_idx: int,
) -> float:
    """Compute log-likelihood of observed frame outcomes.

    LL = Σ [score1_i · log(p_i) + score2_i · log(1 - p_i)]
    where p_i is the predicted frame win probability for player1.

    Args:
        matches: Full match history.
        make_system: Factory function creating a fresh rating system.
        eval_start_idx: Tournament index to start evaluating LL from.
            Earlier tournaments are warm-up only.

    Returns:
        Total log-likelihood (higher = better fit).
    """
    system = make_system()
    tournament_ids = list(matches["tournament_id"].unique())
    ll = 0.0

    for t_idx, tid in enumerate(tournament_ids):
        tourn = matches[matches["tournament_id"] == tid]

        if t_idx >= eval_start_idx:
            # Evaluate on this tournament before updating
            for row in tourn.itertuples(index=False):
                p1 = row.player1
                p2 = row.player2
                s1 = int(row.score1)
                s2 = int(row.score2)

                p = system.predict_frame_win_prob(p1, p2)
                p = np.clip(p, 1e-15, 1 - 1e-15)
                ll += s1 * np.log(p) + s2 * np.log(1 - p)

        # Update system with this tournament
        system.update(tourn)

    return ll


def optimize_elo(
    matches: pd.DataFrame,
    eval_start_idx: int = 0,
    k_bounds: tuple[float, float] = (1.0, 50.0),
    divisor_bounds: tuple[float, float] = (100.0, 800.0),
) -> dict:
    """Find optimal ELO parameters via MLE.

    Optimizes K-factor and divisor to maximize log-likelihood.

    Args:
        matches: Full match history.
        eval_start_idx: Tournament index to start evaluating from.
        k_bounds: Bounds for K-factor search.
        divisor_bounds: Bounds for divisor search.

    Returns:
        Dict with optimal parameters and log-likelihood.
    """

    def neg_ll(params):
        k, divisor = params
        ll = _compute_log_likelihood(
            matches,
            lambda: EloRating(k_factor=k, divisor=divisor),
            eval_start_idx,
        )
        return -ll

    result = minimize(
        neg_ll,
        x0=[8.0, 400.0],
        method="Nelder-Mead",
        bounds=None,  # Nelder-Mead doesn't use bounds directly
        options={"maxiter": 200, "xatol": 0.5, "fatol": 1.0},
    )

    k_opt, divisor_opt = result.x
    # Clip to bounds
    k_opt = np.clip(k_opt, *k_bounds)
    divisor_opt = np.clip(divisor_opt, *divisor_bounds)

    return {
        "k_factor": round(float(k_opt), 2),
        "divisor": round(float(divisor_opt), 2),
        "log_likelihood": -result.fun,
        "converged": result.success,
        "iterations": result.nit,
    }


def optimize_glicko2(
    matches: pd.DataFrame,
    eval_start_idx: int = 0,
    tau_bounds: tuple[float, float] = (0.1, 1.5),
) -> dict:
    """Find optimal Glicko-2 tau parameter via MLE.

    Only optimizes tau (volatility constraint) as it's the most impactful
    parameter. initial_rd and initial_vol have less effect once enough
    matches are processed.

    Args:
        matches: Full match history.
        eval_start_idx: Tournament index to start evaluating from.
        tau_bounds: Bounds for tau search.

    Returns:
        Dict with optimal parameters and log-likelihood.
    """

    def neg_ll(params):
        tau = params[0]
        if tau < tau_bounds[0] or tau > tau_bounds[1]:
            return 1e15
        ll = _compute_log_likelihood(
            matches,
            lambda: Glicko2Rating(tau=tau, default_rating=1500),
            eval_start_idx,
        )
        return -ll

    result = minimize(
        neg_ll,
        x0=[0.5],
        method="Nelder-Mead",
        options={"maxiter": 50, "xatol": 0.01, "fatol": 1.0},
    )

    tau_opt = np.clip(result.x[0], *tau_bounds)

    return {
        "tau": round(float(tau_opt), 3),
        "log_likelihood": -result.fun,
        "converged": result.success,
        "iterations": result.nit,
    }
