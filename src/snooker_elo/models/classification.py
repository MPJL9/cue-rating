"""Match outcome prediction models (binary classification).

Predicts whether player1 or player2 wins a match using features from
both ELO and Glicko-2 rating systems plus player statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    # Verify it actually loads (libomp may be missing on macOS)
    xgb.XGBClassifier()
except Exception:
    xgb = None


# Feature sets
FEATURES_ALL = [
    "player1_elo", "player2_elo",
    "elo_frame_win_rate", "elo_match_win_rate",
    "player1_glicko2", "player2_glicko2",
    "player1_rd", "player2_rd",
    "glicko2_frame_win_rate", "glicko2_match_win_rate",
    "p1_matches_played", "p1_matches_won",
    "p1_frames_played", "p1_frames_won",
    "p2_matches_played", "p2_matches_won",
    "p2_frames_played", "p2_frames_won",
    "p1_frames_played_1_year", "p1_frames_won_1_year",
    "p1_frames_played_3_years", "p1_frames_won_3_years",
    "p2_frames_played_1_year", "p2_frames_won_1_year",
    "p2_frames_played_3_years", "p2_frames_won_3_years",
]

FEATURES_ELO_ONLY = [
    "elo_frame_win_rate", "elo_match_win_rate",
]

FEATURES_GLICKO2_ONLY = [
    "glicko2_frame_win_rate", "glicko2_match_win_rate",
    "player1_rd", "player2_rd",
]

FEATURES_RATINGS_COMBINED = [
    "elo_frame_win_rate", "elo_match_win_rate",
    "glicko2_frame_win_rate", "glicko2_match_win_rate",
    "player1_rd", "player2_rd",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve prediction."""
    df = df.copy()

    # Win rate differences (strongest non-rating features per legacy analysis)
    p1_match_wr = df["p1_matches_won"] / df["p1_matches_played"].replace(0, 1)
    p2_match_wr = df["p2_matches_won"] / df["p2_matches_played"].replace(0, 1)
    df["match_win_rate_diff"] = p1_match_wr - p2_match_wr

    p1_frame_wr = df["p1_frames_won"] / df["p1_frames_played"].replace(0, 1)
    p2_frame_wr = df["p2_frames_won"] / df["p2_frames_played"].replace(0, 1)
    df["frame_win_rate_diff"] = p1_frame_wr - p2_frame_wr

    # 1-year form difference
    p1_1y_wr = df["p1_frames_won_1_year"] / df["p1_frames_played_1_year"].replace(0, 1)
    p2_1y_wr = df["p2_frames_won_1_year"] / df["p2_frames_played_1_year"].replace(0, 1)
    df["form_1y_diff"] = p1_1y_wr - p2_1y_wr

    # 3-year form difference
    p1_3y_wr = df["p1_frames_won_3_years"] / df["p1_frames_played_3_years"].replace(0, 1)
    p2_3y_wr = df["p2_frames_won_3_years"] / df["p2_frames_played_3_years"].replace(0, 1)
    df["form_3y_diff"] = p1_3y_wr - p2_3y_wr

    # ELO rating difference
    df["elo_diff"] = df["player1_elo"] - df["player2_elo"]

    # Glicko-2 rating difference
    df["glicko2_diff"] = df["player1_glicko2"] - df["player2_glicko2"]

    # RD difference (uncertainty gap)
    df["rd_diff"] = df["player1_rd"] - df["player2_rd"]

    # Experience difference
    df["experience_diff"] = df["p1_matches_played"] - df["p2_matches_played"]

    return df


FEATURES_WITH_DERIVED = FEATURES_ALL + [
    "match_win_rate_diff", "frame_win_rate_diff",
    "form_1y_diff", "form_3y_diff",
    "elo_diff", "glicko2_diff", "rd_diff", "experience_diff",
]


def temporal_train_test_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally (no shuffle) for train/test."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_and_evaluate(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    test_fraction: float = 0.2,
) -> dict[str, dict]:
    """Train multiple models on multiple feature sets, return results.

    Args:
        df: Feature DataFrame from generate_features() with derived features added.
        feature_sets: Dict mapping name -> list of feature column names.
        test_fraction: Fraction of data to use for testing.

    Returns:
        Nested dict: {model_name: {feature_set: {metric: value}}}.
    """
    if feature_sets is None:
        feature_sets = {
            "elo_only": FEATURES_ELO_ONLY,
            "glicko2_only": FEATURES_GLICKO2_ONLY,
            "ratings_combined": FEATURES_RATINGS_COMBINED,
            "all_features": FEATURES_WITH_DERIVED,
        }

    train_df, test_df = temporal_train_test_split(df, test_fraction)
    y_train = train_df["match_result"].values
    y_test = test_df["match_result"].values

    results = {}

    for fs_name, features in feature_sets.items():
        # Filter to features that exist in the DataFrame
        available = [f for f in features if f in df.columns]
        if not available:
            continue

        X_train = train_df[available].values.astype(float)
        X_test = test_df[available].values.astype(float)

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Logistic Regression
        lr = LogisticRegression(C=0.1, max_iter=1000)
        lr.fit(X_train_s, y_train)
        lr_probs = lr.predict_proba(X_test_s)[:, 1]
        lr_acc = np.mean((lr_probs > 0.5) == y_test)

        model_key = f"logistic_{fs_name}"
        results[model_key] = {
            "accuracy": lr_acc,
            "features": fs_name,
            "model": "Logistic Regression",
            "n_features": len(available),
        }

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]
        rf_acc = np.mean((rf_probs > 0.5) == y_test)

        model_key = f"random_forest_{fs_name}"
        results[model_key] = {
            "accuracy": rf_acc,
            "features": fs_name,
            "model": "Random Forest",
            "n_features": len(available),
        }

        # Gradient Boosting (sklearn, always available)
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        gb.fit(X_train, y_train)
        gb_probs = gb.predict_proba(X_test)[:, 1]
        gb_acc = np.mean((gb_probs > 0.5) == y_test)

        model_key = f"gradient_boosting_{fs_name}"
        results[model_key] = {
            "accuracy": gb_acc,
            "features": fs_name,
            "model": "Gradient Boosting",
            "n_features": len(available),
        }

        # XGBoost (no scaling needed)
        if xgb is not None:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                verbosity=0,
            )
            xgb_model.fit(X_train, y_train)
            xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
            xgb_acc = np.mean((xgb_probs > 0.5) == y_test)

            model_key = f"xgboost_{fs_name}"
            results[model_key] = {
                "accuracy": xgb_acc,
                "features": fs_name,
                "model": "XGBoost",
                "n_features": len(available),
            }

    return results


def format_results_table(results: dict) -> str:
    """Format model results as a readable table."""
    lines = [
        f"{'Model':<30} {'Features':<20} {'Accuracy':>10} {'N_feat':>8}",
        "-" * 72,
    ]

    # Sort by accuracy descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    for _, info in sorted_results:
        lines.append(
            f"{info['model']:<30} {info['features']:<20} "
            f"{info['accuracy']:>9.4f} {info['n_features']:>8}"
        )

    return "\n".join(lines)
