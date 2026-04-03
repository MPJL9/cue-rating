"""Evaluation metrics for rating systems and prediction models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error


def classification_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Accuracy: predict player1 wins if y_prob > 0.5."""
    y_pred = (y_prob > 0.5).astype(int)
    return accuracy_score(y_true, y_pred)


def match_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Log-loss (cross-entropy): proper scoring rule for probabilistic predictions.

    Lower is better. Heavily penalizes confident wrong predictions.
    """
    # Clip to avoid log(0)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return log_loss(y_true, y_prob, labels=[0, 1])


def match_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of probability predictions.

    Lower is better. Range [0, 1]. 0 = perfect, 0.25 = coin flip.
    """
    return brier_score_loss(y_true, y_prob)


def frame_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error for frame win percentage prediction."""
    return mean_absolute_error(y_true, y_pred)


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration data: bin predicted probabilities vs actual outcomes.

    Returns:
        (bin_centers, actual_rates, bin_counts) — each of length n_bins.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    actual_rates = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            actual_rates[i] = y_true[mask].mean()

    return bin_centers, actual_rates, bin_counts


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE): weighted average of bin calibration errors."""
    bin_centers, actual_rates, bin_counts = calibration_bins(y_true, y_prob, n_bins)
    total = bin_counts.sum()
    if total == 0:
        return 0.0
    ece = np.sum(bin_counts * np.abs(actual_rates - bin_centers)) / total
    return ece


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    frame_true: np.ndarray | None = None,
    frame_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: Binary match outcomes (0 = player1 wins, 1 = player2 wins).
        y_prob: Predicted probability that player1 wins.
        frame_true: Actual frame win proportions (optional).
        frame_pred: Predicted frame win proportions (optional).

    Returns:
        Dict of metric name -> value.
    """
    # For match metrics, y_prob is P(player1 wins), but y_true uses
    # 0=p1 wins, so we need P(outcome) = 1 - y_prob when y_true=0
    # Actually: log_loss/brier expect P(y=1), and y_true=0 means p1 wins.
    # If y_prob = P(p1 wins), then P(y=1) = 1 - y_prob = P(p2 wins).
    p_outcome = 1.0 - y_prob  # P(player2 wins) = P(y_true == 1)

    metrics = {
        "accuracy": classification_accuracy(y_true, p_outcome),
        "log_loss": match_log_loss(y_true, p_outcome),
        "brier_score": match_brier_score(y_true, p_outcome),
        "ece": expected_calibration_error(y_true, p_outcome),
    }

    if frame_true is not None and frame_pred is not None:
        metrics["frame_mae"] = frame_mae(frame_true, frame_pred)

    return metrics
