"""Benchmark our ML model against the World Snooker Tour rankings.

The "always pick higher-ranked player" rule is the simplest reasonable
baseline — it's essentially what casual bettors do, and bookmakers use
official rankings as a major input to their odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

from snooker_elo.data.loader import load_matches
from snooker_elo.features.generator import generate_features
from snooker_elo.models.classification import (
    FEATURES_RATINGS_COMBINED,
    _get_Xy,
    temporal_train_test_split,
)


def load_rankings() -> dict[tuple[int, str], int]:
    """Load year -> {full_name: rank}."""
    df = pd.read_csv("data/raw/world_rankings.csv")
    df["full_name"] = df["First_Name"].astype(str) + " " + df["Surnames"].astype(str)
    df["full_name"] = df["full_name"].str.strip()
    df["Ranking"] = pd.to_numeric(df["Ranking"], errors="coerce")
    df = df.dropna(subset=["Ranking"])
    df = df[df["Ranking"] > 0]  # Drop zero/negative ranks
    df["Ranking"] = df["Ranking"].astype(int)
    return {(int(row.Year), row.full_name): int(row.Ranking) for _, row in df.iterrows()}


def predict_by_ranking(
    matches: pd.DataFrame, rankings: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Predict matches using world rankings.

    Returns: (predictions, probabilities)
    Predictions: 0 = player1 favored, 1 = player2 favored
    Probabilities: P(player2 wins) — 0.6 if player2 strongly higher-ranked,
                   0.5 if no info, 0.4 if player1 higher-ranked
    """
    n = len(matches)
    preds = np.zeros(n, dtype=int)
    probs = np.full(n, 0.5)

    for i, row in enumerate(matches.itertuples(index=False)):
        # Extract year from date string (format: 2019-04-15) or fall back
        date_str = str(getattr(row, "date", ""))
        try:
            year_prev = int(date_str[:4]) - 1
        except (ValueError, IndexError):
            year_prev = 2018  # default
        r1 = rankings.get((year_prev, row.player1))
        r2 = rankings.get((year_prev, row.player2))

        if r1 is None and r2 is None:
            preds[i] = 0
            probs[i] = 0.5
        elif r1 is None:
            preds[i] = 1
            probs[i] = 0.7
        elif r2 is None:
            preds[i] = 0
            probs[i] = 0.3
        elif r1 <= 0 or r2 <= 0:
            preds[i] = 0
            probs[i] = 0.5
        else:
            # Smaller rank = better. Soft probability via log rank ratio.
            log_diff = np.log(r2) - np.log(r1)
            p1_prob = 1 / (1 + np.exp(-log_diff))
            probs[i] = 1 - p1_prob
            preds[i] = 1 if probs[i] > 0.5 else 0

    return preds, probs


def main():
    print("Loading data...")
    matches = load_matches()
    rankings = load_rankings()
    max_ranking_year = max(y for y, _ in rankings.keys())
    print(f"  {len(matches)} matches, {len(rankings)} ranking entries (through {max_ranking_year})")

    # Use the period 2015-2019 for evaluation (when both rankings and modern data overlap)
    # First find the tournaments in that range
    tids_by_year = {}
    for tid, group in matches.groupby("tournament_id", sort=False):
        y = int(group["year"].iloc[0])
        tids_by_year.setdefault(y, []).append(tid)

    eval_tids = []
    for y in range(2015, 2020):  # 2015-2019 inclusive
        eval_tids.extend(tids_by_year.get(y, []))

    all_tids = list(matches["tournament_id"].unique())
    eval_indices = [i for i, t in enumerate(all_tids) if t in set(eval_tids)]
    if not eval_indices:
        print("No matching tournaments found")
        return
    start_idx = min(eval_indices)
    print(f"\nGenerating features for tournaments {start_idx}..end (2015-2019 + warmup)")
    df = generate_features(matches, start_tournament_idx=start_idx)
    # Filter to evaluation period only
    df = df[df["tournament_id"].astype(str).isin(set(map(str, eval_tids)))].reset_index(drop=True)
    print(f"  {len(df)} feature rows in evaluation period")

    # Train/test split temporally within the eval period
    train_df, test_df = temporal_train_test_split(df, test_fraction=0.3)
    y_train = train_df["match_result"].values
    y_test = test_df["match_result"].values
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # ── Baseline 1: Pick higher-ranked player ──
    rank_preds, rank_probs = predict_by_ranking(test_df, rankings)
    rank_acc = (rank_preds == y_test).mean()
    rank_probs_clipped = np.clip(rank_probs, 1e-15, 1 - 1e-15)
    rank_ll = log_loss(y_test, rank_probs_clipped, labels=[0, 1])
    rank_brier = brier_score_loss(y_test, rank_probs_clipped)
    def _year(d):
        try:
            return int(str(d)[:4]) - 1
        except (ValueError, IndexError):
            return 2018
    n_with_rankings = sum(
        1 for _, row in test_df.iterrows()
        if (_year(row["date"]), row["player1"]) in rankings
        and (_year(row["date"]), row["player2"]) in rankings
    )

    # ── Baseline 2: Pure ELO threshold ──
    elo_probs = 1.0 - test_df["elo_match_win_rate"].values
    elo_preds = (elo_probs > 0.5).astype(int)
    elo_acc = (elo_preds == y_test).mean()
    elo_ll = log_loss(y_test, np.clip(elo_probs, 1e-15, 1 - 1e-15), labels=[0, 1])
    elo_brier = brier_score_loss(y_test, np.clip(elo_probs, 1e-15, 1 - 1e-15))

    # ── Our best model ──
    X_train, _ = _get_Xy(train_df, FEATURES_RATINGS_COMBINED)
    X_test, _ = _get_Xy(test_df, FEATURES_RATINGS_COMBINED)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_preds = (gb_probs > 0.5).astype(int)
    gb_acc = (gb_preds == y_test).mean()
    gb_ll = log_loss(y_test, gb_probs, labels=[0, 1])
    gb_brier = brier_score_loss(y_test, gb_probs)

    # ── Coin flip ──
    coin_acc = 0.5
    coin_ll = log_loss(y_test, np.full(len(y_test), 0.5), labels=[0, 1])
    coin_brier = 0.25

    # ── Print results ──
    print(f"\n{'=' * 70}")
    print(f"Test set: {len(test_df)} matches")
    print(f"World rankings cover {n_with_rankings}/{len(test_df)} matches ({n_with_rankings/len(test_df)*100:.0f}%)")
    print(f"{'=' * 70}\n")

    print(f"{'Method':<32} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10}")
    print("-" * 64)
    print(f"{'Coin flip':<32} {coin_acc:>10.4f} {coin_ll:>10.4f} {coin_brier:>10.4f}")
    print(f"{'World Rankings (rank diff)':<32} {rank_acc:>10.4f} {rank_ll:>10.4f} {rank_brier:>10.4f}")
    print(f"{'Pure ELO (threshold)':<32} {elo_acc:>10.4f} {elo_ll:>10.4f} {elo_brier:>10.4f}")
    print(f"{'Our model (GB + 6 features)':<32} {gb_acc:>10.4f} {gb_ll:>10.4f} {gb_brier:>10.4f}")
    print()

    # Improvement over rankings
    delta_acc = gb_acc - rank_acc
    delta_ll = rank_ll - gb_ll  # lower is better, so positive = improvement
    print(f"Our model vs World Rankings:")
    print(f"  Accuracy:  +{delta_acc*100:.1f} percentage points")
    print(f"  Log Loss:  -{delta_ll:.4f} ({delta_ll/rank_ll*100:.1f}% reduction)")

    # Save results
    results = {
        "test_size": int(len(test_df)),
        "ranking_coverage": int(n_with_rankings),
        "methods": {
            "coin_flip": {"accuracy": float(coin_acc), "log_loss": float(coin_ll), "brier": float(coin_brier)},
            "world_rankings": {"accuracy": float(rank_acc), "log_loss": float(rank_ll), "brier": float(rank_brier)},
            "pure_elo": {"accuracy": float(elo_acc), "log_loss": float(elo_ll), "brier": float(elo_brier)},
            "our_model": {"accuracy": float(gb_acc), "log_loss": float(gb_ll), "brier": float(gb_brier)},
        },
    }
    import json
    with open("data/processed/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to data/processed/benchmark_results.json")


if __name__ == "__main__":
    main()
