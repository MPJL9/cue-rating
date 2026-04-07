"""Abstract base class for rating systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PlayerState:
    """Per-player state shared across rating systems."""

    rating: float = 1000.0
    matches_played: int = 0
    matches_won: int = 0
    frames_played: int = 0
    frames_won: int = 0
    # Time-windowed stats
    matches_played_3y: int = 0
    matches_won_3y: int = 0
    frames_played_3y: int = 0
    frames_won_3y: int = 0
    matches_played_1y: int = 0
    matches_won_1y: int = 0
    frames_played_1y: int = 0
    frames_won_1y: int = 0
    # Rating history: list of (tournament_id, rating) snapshots
    rating_history: list = field(default_factory=list)
    # Momentum: rating N matches ago (for computing rating change)
    prev_rating: float = 0.0
    # Last match index (for computing inactivity)
    last_match_idx: int = 0


class RatingSystem(ABC):
    """Abstract interface for snooker rating systems (ELO, Glicko-2, etc.)."""

    def __init__(self, default_rating: float = 1000.0):
        self.default_rating = default_rating
        self.players: dict[str, PlayerState] = {}

    def _ensure_player(self, name: str) -> PlayerState:
        """Get or create a player's state."""
        if name not in self.players:
            self.players[name] = self._new_player_state()
        return self.players[name]

    def _new_player_state(self) -> PlayerState:
        """Create a new player state. Override for systems with extra fields."""
        return PlayerState(rating=self.default_rating)

    @abstractmethod
    def update(self, matches: pd.DataFrame) -> None:
        """Process matches chronologically and update all player ratings/stats.

        Args:
            matches: DataFrame with columns [player1, player2, score1, score2,
                     best_of, tournament_id, date, year].
                     player1 is always the winner (score1 >= score2).
        """

    def predict_frame_win_prob(self, player1: str, player2: str) -> float:
        """Predicted probability that player1 wins a frame against player2."""
        r1 = self._ensure_player(player1).rating
        r2 = self._ensure_player(player2).rating
        return self._frame_win_prob(r1, r2)

    @abstractmethod
    def _frame_win_prob(self, r1: float, r2: float) -> float:
        """Compute frame win probability from two ratings."""

    @staticmethod
    def match_win_prob(p: float, best_of: int) -> float:
        """Convert frame win probability to match win probability via binomial.

        Args:
            p: Probability player1 wins each frame.
            best_of: Maximum number of frames (e.g. 17 for first-to-9).

        Returns:
            Probability player1 wins the match.
        """
        from math import comb

        best_of = int(best_of)
        if best_of < 1:
            return p  # Degenerate case: single frame
        win_cond = (best_of + 1) // 2
        if win_cond < 1:
            return p
        q = 0.0
        for total in range(win_cond, best_of + 1):
            score1 = win_cond
            score2 = total - score1
            q += comb(total - 1, score1 - 1) * (p ** (score1 - 1)) * ((1 - p) ** score2) * p
        return q

    def predict_match_win_prob(self, player1: str, player2: str, best_of: int) -> float:
        """Predicted probability that player1 wins a match against player2."""
        p = self.predict_frame_win_prob(player1, player2)
        return self.match_win_prob(p, best_of)

    def get_ratings(self) -> pd.Series:
        """Return current ratings as a pandas Series, sorted descending."""
        data = {name: state.rating for name, state in self.players.items()}
        s = pd.Series(data, dtype=float)
        return s.sort_values(ascending=False)

    def get_player_stats(self) -> pd.DataFrame:
        """Return a DataFrame of all player statistics."""
        rows = {}
        for name, s in self.players.items():
            rows[name] = {
                "elo_rating": round(s.rating),
                "matches_played": s.matches_played,
                "matches_won": s.matches_won,
                "matches_win_rate": s.matches_won / s.matches_played if s.matches_played else 0.0,
                "frames_played": s.frames_played,
                "frames_won": s.frames_won,
                "frames_win_rate": s.frames_won / s.frames_played if s.frames_played else 0.0,
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def snapshot(self) -> dict[str, PlayerState]:
        """Return a shallow copy of all player states (for feature generation)."""
        from copy import copy

        return {name: copy(state) for name, state in self.players.items()}

    def predict_batch(
        self, matches: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict outcomes for a batch of matches.

        Args:
            matches: DataFrame with columns [player1, player2, best_of].

        Returns:
            (predictions, frame_win_rates, match_win_rates) as numpy arrays.
            predictions[i] = True means player2 is predicted to win.
        """
        n = len(matches)
        frame_probs = np.empty(n)
        match_probs = np.empty(n)

        for i, row in enumerate(matches.itertuples(index=False)):
            p1, p2, bo = row.player1, row.player2, int(row.best_of)
            fp = self.predict_frame_win_prob(p1, p2)
            frame_probs[i] = fp
            match_probs[i] = self.match_win_prob(fp, bo)

        predictions = frame_probs <= 0.5
        return predictions, frame_probs, match_probs
