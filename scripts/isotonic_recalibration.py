"""Isotonic recalibration experiment.

A model can be accurate but poorly calibrated, or well-calibrated but
inaccurate. Calibration measures whether predicted probabilities match
empirical frequencies — when the model says "70% probability", does
the favored player actually win ~70% of the time?

Isotonic regression is a non-parametric monotonic mapping from raw
predicted probabilities to recalibrated probabilities, fit on a held-out
calibration set. It can only improve calibration if the original model
is mis-calibrated.

We test isotonic recalibration on three baselines:
1. Pure ELO match win probability
2. Pure Glicko-2 match win probability
3. Gradient Boosting on the 6-feature ratings_combined set

For each, we report:
- ECE before and after recalibration
- Brier score before and after
- Whether it actually helped
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from snooker_elo.data.loader import load_matches
from snooker_elo.evaluation.metrics import expected_calibration_error
from snooker_elo.features.generator import generate_features
from snooker_elo.models.classification import (
    FEATURES_RATINGS_COMBINED,
    _get_Xy,
    temporal_train_test_split,
)


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    pred = (y_prob > 0.5).astype(int)
    return {
        "name": name,
        "accuracy": float((pred == y_true).mean()),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
    }


def recalibrate(probs_train, y_train, probs_test):
    """Fit isotonic regression on train probs vs train y, apply to test probs."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(probs_train, y_train)
    return iso.transform(probs_test)


def main():
    print("Loading data and generating features...")
    matches = load_matches()
    n_tourns = len(list(matches["tournament_id"].unique()))
    df = generate_features(matches, start_tournament_idx=n_tourns - 300)

    # Three-way temporal split: train -> calibration -> test
    # 60% train, 20% calibration, 20% test
    n = len(df)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)
    train_df = df.iloc[:train_end].copy()
    calib_df = df.iloc[train_end:calib_end].copy()
    test_df = df.iloc[calib_end:].copy()
    print(f"  Train: {len(train_df)}, Calibration: {len(calib_df)}, Test: {len(test_df)}")

    y_calib = calib_df["match_result"].values
    y_test = test_df["match_result"].values

    results = []

    # ── Pure ELO ──
    elo_calib = 1.0 - calib_df["elo_match_win_rate"].values
    elo_test = 1.0 - test_df["elo_match_win_rate"].values
    results.append(evaluate("Pure ELO (raw)", y_test, elo_test))

    elo_test_recal = recalibrate(elo_calib, y_calib, elo_test)
    results.append(evaluate("Pure ELO (isotonic)", y_test, elo_test_recal))

    # ── Pure Glicko-2 ──
    g2_calib = 1.0 - calib_df["glicko2_match_win_rate"].values
    g2_test = 1.0 - test_df["glicko2_match_win_rate"].values
    results.append(evaluate("Pure Glicko-2 (raw)", y_test, g2_test))

    g2_test_recal = recalibrate(g2_calib, y_calib, g2_test)
    results.append(evaluate("Pure Glicko-2 (isotonic)", y_test, g2_test_recal))

    # ── Gradient Boosting on 6 features ──
    print("Training Gradient Boosting...")
    X_train, _ = _get_Xy(train_df, FEATURES_RATINGS_COMBINED)
    y_train_gb = train_df["match_result"].values
    X_calib, _ = _get_Xy(calib_df, FEATURES_RATINGS_COMBINED)
    X_test, _ = _get_Xy(test_df, FEATURES_RATINGS_COMBINED)

    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gb.fit(X_train, y_train_gb)

    gb_calib = gb.predict_proba(X_calib)[:, 1]
    gb_test = gb.predict_proba(X_test)[:, 1]
    results.append(evaluate("GB 6-feature (raw)", y_test, gb_test))

    gb_test_recal = recalibrate(gb_calib, y_calib, gb_test)
    results.append(evaluate("GB 6-feature (isotonic)", y_test, gb_test_recal))

    # ── Print results ──
    print(f"\n{'=' * 70}")
    print(f"{'Model':<30} {'Accuracy':>10} {'Brier':>10} {'ECE':>10}")
    print("-" * 64)
    for r in results:
        print(f"{r['name']:<30} {r['accuracy']:>10.4f} {r['brier']:>10.4f} {r['ece']:>10.4f}")
    print("=" * 70)

    # Compare improvement
    print("\nIsotonic recalibration impact:")
    pairs = [
        ("Pure ELO", results[0], results[1]),
        ("Pure Glicko-2", results[2], results[3]),
        ("GB 6-feature", results[4], results[5]),
    ]
    for name, raw, cal in pairs:
        d_brier = raw["brier"] - cal["brier"]
        d_ece = raw["ece"] - cal["ece"]
        print(f"  {name:<20} ΔBrier: {d_brier:+.4f}  ΔECE: {d_ece:+.4f}  "
              f"({'helped' if d_ece > 0 else 'hurt' if d_ece < -0.001 else 'no change'})")

    with open("data/processed/isotonic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to data/processed/isotonic_results.json")


if __name__ == "__main__":
    main()
