"""Systematic comparison between rating systems.

Evaluates ELO vs Glicko-2 using temporal expanding-window cross-validation
with multiple metrics (accuracy, log-loss, Brier score, calibration).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from snooker_elo.evaluation.metrics import compute_all_metrics
from snooker_elo.ratings.elo import EloRating
from snooker_elo.ratings.glicko2 import Glicko2Rating


def compare_systems(
    matches: pd.DataFrame,
    elo_params: dict | None = None,
    glicko2_params: dict | None = None,
    eval_start_idx: int = 0,
) -> dict:
    """Run a full comparison between ELO and Glicko-2 on match data.

    Uses temporal evaluation: trains on tournaments [0..N-1], evaluates on N,
    then rolls forward. This simulates real-world prediction.

    Args:
        matches: Full match history ordered chronologically.
        elo_params: EloRating constructor kwargs (default: k=8, divisor=400).
        glicko2_params: Glicko2Rating constructor kwargs (default: tau=0.5).
        eval_start_idx: Tournament index to start evaluating from.

    Returns:
        Dict with:
        - 'elo_metrics': aggregate metrics for ELO
        - 'glicko2_metrics': aggregate metrics for Glicko-2
        - 'per_tournament': per-tournament metric breakdown
        - 'elo_predictions': raw predictions for further analysis
        - 'glicko2_predictions': raw predictions for further analysis
    """
    if elo_params is None:
        elo_params = {"k_factor": 8, "divisor": 400}
    if glicko2_params is None:
        glicko2_params = {"tau": 0.5, "default_rating": 1500}

    elo = EloRating(**elo_params)
    g2 = Glicko2Rating(**glicko2_params)

    tournament_ids = list(matches["tournament_id"].unique())

    # Collect predictions
    elo_preds = []  # (match_result, elo_frame_prob, elo_match_prob, win_pct)
    g2_preds = []

    per_tournament = []

    for t_idx, tid in enumerate(tournament_ids):
        tourn = matches[matches["tournament_id"] == tid]

        if t_idx >= eval_start_idx:
            # Evaluate before updating
            tourn_elo = []
            tourn_g2 = []

            for row in tourn.itertuples(index=False):
                p1, p2 = row.player1, row.player2
                s1, s2 = int(row.score1), int(row.score2)
                bo = int(row.best_of)
                win_pct = s1 / (s1 + s2)

                # ELO predictions
                elo_fp = elo.predict_frame_win_prob(p1, p2)
                elo_mp = elo.match_win_prob(elo_fp, bo)

                # Glicko-2 predictions
                g2_fp = g2.predict_frame_win_prob(p1, p2)
                g2_mp = g2.match_win_prob(g2_fp, bo)

                # match_result: 0 = player1 wins (always true in raw data)
                elo_preds.append((0, elo_fp, elo_mp, win_pct))
                g2_preds.append((0, g2_fp, g2_mp, win_pct))

                tourn_elo.append((0, elo_fp, elo_mp, win_pct))
                tourn_g2.append((0, g2_fp, g2_mp, win_pct))

            if tourn_elo:
                tourn_elo = np.array(tourn_elo)
                tourn_g2 = np.array(tourn_g2)

                elo_acc = np.mean(tourn_elo[:, 2] > 0.5)  # match_prob > 0.5 for p1
                g2_acc = np.mean(tourn_g2[:, 2] > 0.5)

                per_tournament.append({
                    "tournament_id": tid,
                    "n_matches": len(tourn),
                    "elo_accuracy": elo_acc,
                    "glicko2_accuracy": g2_acc,
                })

        # Update both systems
        elo.update(tourn)
        g2.update(tourn)

    # Aggregate metrics
    elo_arr = np.array(elo_preds)
    g2_arr = np.array(g2_preds)

    if len(elo_arr) == 0:
        return {"elo_metrics": {}, "glicko2_metrics": {}, "per_tournament": []}

    y_true = elo_arr[:, 0].astype(int)  # All zeros (player1 always wins in raw)
    elo_frame_pred = elo_arr[:, 1]
    elo_match_pred = elo_arr[:, 2]
    g2_frame_pred = g2_arr[:, 1]
    g2_match_pred = g2_arr[:, 2]
    actual_win_pct = elo_arr[:, 3]

    elo_metrics = compute_all_metrics(
        y_true, elo_match_pred,
        frame_true=actual_win_pct, frame_pred=elo_frame_pred,
    )
    g2_metrics = compute_all_metrics(
        y_true, g2_match_pred,
        frame_true=actual_win_pct, frame_pred=g2_frame_pred,
    )

    return {
        "elo_metrics": elo_metrics,
        "glicko2_metrics": g2_metrics,
        "per_tournament": per_tournament,
        "elo_predictions": elo_arr,
        "glicko2_predictions": g2_arr,
    }


def format_comparison_table(results: dict) -> str:
    """Format comparison results as a readable table."""
    elo = results["elo_metrics"]
    g2 = results["glicko2_metrics"]

    lines = [
        "=" * 55,
        f"{'Metric':<25} {'ELO':>12} {'Glicko-2':>12}",
        "-" * 55,
    ]

    metric_names = {
        "accuracy": ("Accuracy", True),      # higher is better
        "log_loss": ("Log Loss", False),      # lower is better
        "brier_score": ("Brier Score", False), # lower is better
        "ece": ("ECE", False),                 # lower is better
        "frame_mae": ("Frame MAE", False),     # lower is better
    }

    for key, (name, higher_better) in metric_names.items():
        if key in elo and key in g2:
            e_val = elo[key]
            g_val = g2[key]
            if higher_better:
                if e_val > g_val:
                    e_str = f"{e_val:.4f} *"
                    g_str = f"{g_val:.4f}"
                else:
                    e_str = f"{e_val:.4f}"
                    g_str = f"{g_val:.4f} *"
            else:
                if e_val < g_val:
                    e_str = f"{e_val:.4f} *"
                    g_str = f"{g_val:.4f}"
                else:
                    e_str = f"{e_val:.4f}"
                    g_str = f"{g_val:.4f} *"

            lines.append(f"{name:<25} {e_str:>12} {g_str:>12}")

    lines.append("=" * 55)
    lines.append("* = better")

    n_matches = len(results.get("elo_predictions", []))
    n_tourns = len(results.get("per_tournament", []))
    lines.append(f"Evaluated on {n_matches} matches across {n_tourns} tournaments")

    return "\n".join(lines)
