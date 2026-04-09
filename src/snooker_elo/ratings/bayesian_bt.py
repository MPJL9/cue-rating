"""Bayesian Bradley-Terry rating system loaded from precomputed posterior.

This system loads MCMC samples from a pickle file (generated offline by
scripts/fit_bayesian_bt.py) and exposes them via the RatingSystem
interface.

Predictions use posterior predictive averaging:
    P(i beats j in a frame) = mean over samples of sigmoid(s_i^(k) - s_j^(k))
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from snooker_elo.ratings.base import PlayerState, RatingSystem


@dataclass
class BayesianPlayerState(PlayerState):
    """Player state for Bayesian BT — stores posterior samples."""

    skill_samples: np.ndarray = field(default_factory=lambda: np.zeros(0))
    skill_mean: float = 0.0
    skill_std: float = 0.0


class BayesianBTRating(RatingSystem):
    """Bayesian Bradley-Terry rating system.

    Loads precomputed posterior samples from a pickle file and exposes
    them via the standard RatingSystem interface. Predictions use
    posterior predictive averaging — averaging the win probability
    across all posterior samples for proper uncertainty propagation.

    Players not in the posterior get default (skill=0) — they have
    no rating data.
    """

    def __init__(
        self,
        posterior_path: str | Path | None = None,
        default_rating: float = 0.0,
    ):
        super().__init__(default_rating=default_rating)
        self.posterior_path = posterior_path
        self._sample_matrix: np.ndarray | None = None  # (n_samples, n_players)
        self._player_to_col: dict[str, int] = {}

    def _new_player_state(self) -> BayesianPlayerState:
        return BayesianPlayerState(rating=self.default_rating)

    def load_posterior(self, path: str | Path | None = None) -> None:
        """Load posterior samples from a pickle file."""
        if path is None:
            path = self.posterior_path
        if path is None:
            raise ValueError("posterior_path must be provided")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._sample_matrix = data["posterior_samples"]  # (n_samples, n_players)
        players = data["players"]
        self._player_to_col = {p: i for i, p in enumerate(players)}

        # Build player states
        means = self._sample_matrix.mean(axis=0)
        stds = self._sample_matrix.std(axis=0)
        for i, name in enumerate(players):
            self.players[name] = BayesianPlayerState(
                rating=float(means[i]),
                skill_mean=float(means[i]),
                skill_std=float(stds[i]),
                skill_samples=self._sample_matrix[:, i],
            )

    def update(self, matches: pd.DataFrame) -> None:
        """No-op: this rating system is fit offline via MCMC."""
        # The Bayesian fit happens in scripts/fit_bayesian_bt.py
        # This update method exists only to satisfy the ABC.

    def _frame_win_prob(self, r1: float, r2: float) -> float:
        """Point-estimate frame win probability via the BT formula."""
        return 1.0 / (1.0 + math.exp(-(r1 - r2)))

    def predict_frame_win_prob(self, player1: str, player2: str) -> float:
        """Posterior predictive frame win probability.

        Averages sigmoid(s1 - s2) over all posterior samples.
        For players not in the posterior, uses the default skill (0).
        """
        col1 = self._player_to_col.get(player1)
        col2 = self._player_to_col.get(player2)

        if self._sample_matrix is None or (col1 is None and col2 is None):
            # No data for either player
            return 0.5

        if col1 is None:
            # Only player2 has samples; treat player1 as average (skill=0)
            s2 = self._sample_matrix[:, col2]
            probs = 1.0 / (1.0 + np.exp(s2))  # P(0 > s2)
            return float(probs.mean())

        if col2 is None:
            s1 = self._sample_matrix[:, col1]
            probs = 1.0 / (1.0 + np.exp(-s1))  # P(s1 > 0)
            return float(probs.mean())

        # Both have samples — proper posterior predictive
        s1 = self._sample_matrix[:, col1]
        s2 = self._sample_matrix[:, col2]
        probs = 1.0 / (1.0 + np.exp(-(s1 - s2)))
        return float(probs.mean())

    def predict_frame_win_prob_with_uncertainty(
        self, player1: str, player2: str
    ) -> tuple[float, float, float]:
        """Return (mean, q025, q975) of the posterior over win probability."""
        col1 = self._player_to_col.get(player1)
        col2 = self._player_to_col.get(player2)

        if self._sample_matrix is None or col1 is None or col2 is None:
            return (0.5, 0.5, 0.5)

        s1 = self._sample_matrix[:, col1]
        s2 = self._sample_matrix[:, col2]
        probs = 1.0 / (1.0 + np.exp(-(s1 - s2)))
        return (float(probs.mean()), float(np.quantile(probs, 0.025)),
                float(np.quantile(probs, 0.975)))

    def get_player_stats(self) -> pd.DataFrame:
        """Return a summary DataFrame with skill mean, std, and credible interval."""
        rows = {}
        for name, state in self.players.items():
            if not isinstance(state, BayesianPlayerState):
                continue
            samples = state.skill_samples
            rows[name] = {
                "skill_mean": round(state.skill_mean, 4),
                "skill_std": round(state.skill_std, 4),
                "skill_q025": round(float(np.quantile(samples, 0.025)), 4),
                "skill_q975": round(float(np.quantile(samples, 0.975)), 4),
            }
        df = pd.DataFrame.from_dict(rows, orient="index")
        return df.sort_values("skill_mean", ascending=False)
