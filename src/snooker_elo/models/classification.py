"""Match outcome prediction models (binary classification).

Predicts whether player1 or player2 wins a match using features from
both ELO and Glicko-2 rating systems plus player statistics.
Supports multiple feature combos and dimensionality reduction (PCA).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    # Verify it actually loads (libomp may be missing on macOS)
    xgb.XGBClassifier()
except Exception:
    xgb = None


# ── Feature sets ──────────────────────────────────────────────────────────

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

# Clean non-redundant features (differences only, no raw counts)
FEATURES_CLEAN = [
    "elo_frame_win_rate", "elo_match_win_rate",
    "glicko2_frame_win_rate", "glicko2_match_win_rate",
    "player1_rd", "player2_rd",
    "match_wr_diff", "frame_wr_diff",
    "form_1y_diff", "form_3y_diff",
    "experience_diff",
    "h2h_advantage", "h2h_matches",
    "momentum_diff", "inactivity_diff",
]

# All raw features (including redundant ones for PCA)
FEATURES_ALL_RAW = [
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
    "match_wr_diff", "frame_wr_diff",
    "form_1y_diff", "form_3y_diff",
    "experience_diff",
    "h2h_advantage", "h2h_matches",
    "momentum_diff", "inactivity_diff",
]


def temporal_train_test_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally (no shuffle) for train/test."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _get_Xy(df: pd.DataFrame, features: list[str]):
    """Extract feature matrix and handle NaN/inf."""
    available = [f for f in features if f in df.columns]
    X = df[available].values.astype(float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    return X, available


def train_and_evaluate(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    test_fraction: float = 0.2,
) -> dict[str, dict]:
    """Train multiple models on multiple feature sets, return results.

    Includes:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - PCA + Logistic Regression (on the all-raw set)
    - XGBoost (if available)

    Args:
        df: Feature DataFrame from generate_features().
        feature_sets: Dict mapping name -> list of feature column names.
        test_fraction: Fraction of data for testing.

    Returns:
        Nested dict: {model_key: {metric: value}}.
    """
    if feature_sets is None:
        feature_sets = {
            "elo_only": FEATURES_ELO_ONLY,
            "glicko2_only": FEATURES_GLICKO2_ONLY,
            "ratings_combined": FEATURES_RATINGS_COMBINED,
            "clean_15": FEATURES_CLEAN,
            "all_raw": FEATURES_ALL_RAW,
        }

    train_df, test_df = temporal_train_test_split(df, test_fraction)
    y_train = train_df["match_result"].values
    y_test = test_df["match_result"].values

    results = {}

    for fs_name, features in feature_sets.items():
        X_train, available = _get_Xy(train_df, features)
        X_test, _ = _get_Xy(test_df, features)
        n_feat = len(available)

        if n_feat == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ── Logistic Regression ──
        lr = LogisticRegression(C=0.1, max_iter=1000)
        lr.fit(X_train_s, y_train)
        lr_acc = np.mean((lr.predict_proba(X_test_s)[:, 1] > 0.5) == y_test)
        results[f"logistic_{fs_name}"] = {
            "accuracy": lr_acc, "features": fs_name,
            "model": "Logistic Regression", "n_features": n_feat,
        }

        # ── Random Forest ──
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_acc = np.mean((rf.predict_proba(X_test)[:, 1] > 0.5) == y_test)
        results[f"rf_{fs_name}"] = {
            "accuracy": rf_acc, "features": fs_name,
            "model": "Random Forest", "n_features": n_feat,
        }

        # ── Gradient Boosting ──
        gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        gb.fit(X_train, y_train)
        gb_acc = np.mean((gb.predict_proba(X_test)[:, 1] > 0.5) == y_test)
        results[f"gb_{fs_name}"] = {
            "accuracy": gb_acc, "features": fs_name,
            "model": "Gradient Boosting", "n_features": n_feat,
        }

        # ── PCA + Logistic Regression (only for sets with >6 features) ──
        if n_feat > 6:
            for n_comp in [5, 10]:
                if n_comp >= n_feat:
                    continue
                pca_pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_comp)),
                    ("lr", LogisticRegression(C=0.1, max_iter=1000)),
                ])
                pca_pipe.fit(X_train, y_train)
                pca_acc = np.mean((pca_pipe.predict_proba(X_test)[:, 1] > 0.5) == y_test)
                results[f"pca{n_comp}_lr_{fs_name}"] = {
                    "accuracy": pca_acc, "features": f"{fs_name} (PCA-{n_comp})",
                    "model": f"PCA({n_comp}) + LR", "n_features": n_comp,
                }

        # ── XGBoost ──
        if xgb is not None:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", verbosity=0,
            )
            xgb_model.fit(X_train, y_train)
            xgb_acc = np.mean((xgb_model.predict_proba(X_test)[:, 1] > 0.5) == y_test)
            results[f"xgb_{fs_name}"] = {
                "accuracy": xgb_acc, "features": fs_name,
                "model": "XGBoost", "n_features": n_feat,
            }

    return results


def format_results_table(results: dict) -> str:
    """Format model results as a readable table."""
    lines = [
        f"{'Model':<25} {'Features':<25} {'Accuracy':>10} {'N_feat':>8}",
        "-" * 72,
    ]
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for _, info in sorted_results:
        lines.append(
            f"{info['model']:<25} {info['features']:<25} "
            f"{info['accuracy']:>9.4f} {info['n_features']:>8}"
        )
    return "\n".join(lines)
