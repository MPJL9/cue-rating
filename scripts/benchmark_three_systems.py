"""Benchmark ELO vs Glicko-2 vs Bayesian Bradley-Terry on the same test set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from snooker_elo.data.loader import load_matches
from snooker_elo.ratings.bayesian_bt import BayesianBTRating
from snooker_elo.ratings.elo import EloRating
from snooker_elo.ratings.glicko2 import Glicko2Rating


def evaluate(name, system, test_matches, has_bayesian=False):
    """Evaluate a rating system on test matches.

    Returns: (accuracy, log_loss, brier, n_evaluated)
    """
    y_true, y_prob = [], []
    for _, row in test_matches.iterrows():
        p1, p2 = row["player1"], row["player2"]
        bo = int(row["best_of"])

        # Skip if not in this system's player set
        if has_bayesian:
            if p1 not in system._player_to_col or p2 not in system._player_to_col:
                continue

        fp = system.predict_frame_win_prob(p1, p2)
        mp = system.match_win_prob(fp, bo)

        # Player1 always won in raw data
        y_true.append(0)
        y_prob.append(1.0 - mp)  # P(player2 wins)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(y_true) == 0:
        return 0, 0, 0, 0

    pred = (y_prob > 0.5).astype(int)
    acc = (pred == y_true).mean()
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])
    brier = brier_score_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))

    return acc, ll, brier, len(y_true)


def main():
    print("Loading data...")
    matches = load_matches()

    # Use last 2 years for test (matches what Bayesian was trained on)
    max_year = int(matches["year"].max())
    cutoff_train = max_year - 2

    train = matches[matches["year"] < cutoff_train].copy()
    test = matches[matches["year"] >= cutoff_train].copy()
    test = test[test["score1"] != test["score2"]]  # drop draws

    print(f"  Train: {len(train)}, Test: {len(test)}")

    # ── Build ELO ──
    print("Building ELO (full history)...")
    elo = EloRating(k_factor=9.77, divisor=327.15)
    elo.update(train)

    # ── Build Glicko-2 ──
    print("Building Glicko-2 (full history)...")
    g2 = Glicko2Rating(tau=1.488, default_rating=1500)
    g2.update(train)

    # ── Load Bayesian BT ──
    print("Loading Bayesian BT posterior...")
    bayes = BayesianBTRating()
    bayes.load_posterior("data/raw/bayesian_bt_cache.pkl")
    print(f"  Posterior covers {len(bayes._player_to_col)} players")

    # Filter test to matches where both players are in the Bayesian set
    # (for fair comparison — all systems evaluated on same matches)
    bp = set(bayes._player_to_col.keys())
    test_common = test[test["player1"].isin(bp) & test["player2"].isin(bp)].copy()
    print(f"  Common test matches: {len(test_common)} (both players in Bayesian set)")

    print(f"\n{'=' * 70}")
    print(f"{'System':<25} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} {'N':>8}")
    print("-" * 70)
    for name, sys, has_b in [
        ("ELO", elo, False),
        ("Glicko-2", g2, False),
        ("Bayesian BT", bayes, True),
    ]:
        acc, ll, brier, n = evaluate(name, sys, test_common, has_bayesian=False)
        print(f"{name:<25} {acc:>10.4f} {ll:>10.4f} {brier:>10.4f} {n:>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
