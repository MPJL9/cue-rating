"""ELO rating system for snooker, optimized with dict-based state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from snooker_elo.ratings.base import PlayerState, RatingSystem


class EloRating(RatingSystem):
    """Frame-weighted ELO rating system for professional snooker.

    The key adaptation from chess ELO:
    - Rating reflects *frame* win probability, not match win probability.
    - Updates are weighted by total frames played in the match.
    - Match win probability is derived via binomial formula.

    Update rule:
        R_new = R + K * (score1 + score2) * (S - E)
    where:
        E = 1 / (1 + exp((R_opponent - R) / divisor))
        S = frames_won / total_frames
    """

    def __init__(
        self,
        k_factor: float = 8.0,
        divisor: float = 400.0,
        default_rating: float = 1000.0,
    ):
        super().__init__(default_rating=default_rating)
        self.k_factor = k_factor
        self.divisor = divisor

    def _frame_win_prob(self, r1: float, r2: float) -> float:
        return 1.0 / (1.0 + np.exp((r2 - r1) / self.divisor))

    @staticmethod
    def _calculate_new_elo(
        r1: float, r2: float, score1: int, score2: int, k: float, divisor: float
    ) -> tuple[float, float]:
        """Compute new ELO ratings after a match.

        Returns (r1_new, r2_new).
        """
        total = score1 + score2
        e1 = 1.0 / (1.0 + np.exp((r2 - r1) / divisor))
        s1 = score1 / total
        r1_new = round(r1 + k * total * (s1 - e1))
        r2_new = round(r2 + k * total * (e1 - s1))
        return r1_new, r2_new

    def update(self, matches: pd.DataFrame) -> None:
        """Process all matches chronologically and update player states.

        Uses dict-based state for O(1) lookups instead of DataFrame .loc[].

        Args:
            matches: DataFrame with columns [player1, player2, score1, score2,
                     best_of, tournament_id, date, year].
        """
        current_year = int(matches["year"].max())
        k = self.k_factor
        divisor = self.divisor
        default_rating = self.default_rating
        players = self.players

        # Pre-register all players
        all_names = pd.concat([matches["player1"], matches["player2"]]).unique()
        for name in all_names:
            if name not in players:
                players[name] = PlayerState(rating=default_rating, prev_rating=default_rating)

        # Track match index for inactivity and tournament boundaries for history
        match_idx = 0
        current_tid = None

        # Main loop: iterate via itertuples for speed
        for row in matches.itertuples(index=False):
            p1_name = row.player1
            p2_name = row.player2
            score1 = int(row.score1)
            score2 = int(row.score2)
            year = int(row.year)
            tid = str(row.tournament_id)

            # Record rating snapshot at tournament boundaries
            if tid != current_tid:
                current_tid = tid
                for pname in (p1_name, p2_name):
                    p = players[pname]
                    if not p.rating_history or p.rating_history[-1][0] != tid:
                        p.rating_history.append((tid, round(p.rating), year))

            p1 = players[p1_name]
            p2 = players[p2_name]

            # Save pre-match rating for momentum tracking
            if p1.matches_played > 0 and p1.matches_played % 10 == 0:
                p1.prev_rating = p1.rating
            if p2.matches_played > 0 and p2.matches_played % 10 == 0:
                p2.prev_rating = p2.rating

            # Update ELO ratings
            r1_new, r2_new = self._calculate_new_elo(
                p1.rating, p2.rating, score1, score2, k, divisor
            )
            p1.rating = r1_new
            p2.rating = r2_new

            # Track match index for inactivity
            match_idx += 1
            p1.last_match_idx = match_idx
            p2.last_match_idx = match_idx

            # Update career stats
            total_frames = score1 + score2
            p1.matches_played += 1
            p2.matches_played += 1
            p1.matches_won += 1  # player1 is always the winner in raw data

            p1.frames_played += total_frames
            p2.frames_played += total_frames
            p1.frames_won += score1
            p2.frames_won += score2

            # Update time-windowed stats
            if current_year - year < 3:
                p1.matches_played_3y += 1
                p2.matches_played_3y += 1
                p1.matches_won_3y += 1
                p1.frames_played_3y += total_frames
                p2.frames_played_3y += total_frames
                p1.frames_won_3y += score1
                p2.frames_won_3y += score2

            if current_year - year < 1:
                p1.matches_played_1y += 1
                p2.matches_played_1y += 1
                p1.matches_won_1y += 1
                p1.frames_played_1y += total_frames
                p2.frames_played_1y += total_frames
                p1.frames_won_1y += score1
                p2.frames_won_1y += score2

        # Record final ratings
        for pname, p in players.items():
            if p.rating_history and p.rating_history[-1][0] != "final":
                p.rating_history.append(("final", round(p.rating), current_year))

    def generate_match_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Generate feature DataFrame for a set of matches using current state.

        This snapshots the current player states and generates features.
        Call this BEFORE updating with these matches (to avoid data leakage).

        Args:
            matches: DataFrame with columns [player1, player2, best_of].

        Returns:
            DataFrame with ELO ratings, predictions, and player stats for each match.
        """
        players = self.players
        default_rating = self.default_rating

        rows = []
        for row in matches.itertuples(index=False):
            p1_name = row.player1
            p2_name = row.player2
            best_of = int(row.best_of)

            p1 = players.get(p1_name, PlayerState(rating=default_rating))
            p2 = players.get(p2_name, PlayerState(rating=default_rating))

            fp = self._frame_win_prob(p1.rating, p2.rating)
            mp = self.match_win_prob(fp, best_of)

            rows.append(
                {
                    "player1": p1_name,
                    "player2": p2_name,
                    "best_of": best_of,
                    "player1_elo": round(p1.rating),
                    "player2_elo": round(p2.rating),
                    "elo_match_win_rate": mp,
                    "elo_frame_win_rate": fp,
                    "p1_matches_played": p1.matches_played,
                    "p1_matches_won": p1.matches_won,
                    "p1_frames_played": p1.frames_played,
                    "p1_frames_won": p1.frames_won,
                    "p2_matches_played": p2.matches_played,
                    "p2_matches_won": p2.matches_won,
                    "p2_frames_played": p2.frames_played,
                    "p2_frames_won": p2.frames_won,
                    "p1_frames_played_1_year": p1.frames_played_1y,
                    "p1_frames_won_1_year": p1.frames_won_1y,
                    "p1_frames_played_3_years": p1.frames_played_3y,
                    "p1_frames_won_3_years": p1.frames_won_3y,
                    "p2_frames_played_1_year": p2.frames_played_1y,
                    "p2_frames_won_1_year": p2.frames_won_1y,
                    "p2_frames_played_3_years": p2.frames_played_3y,
                    "p2_frames_won_3_years": p2.frames_won_3y,
                }
            )

        return pd.DataFrame(rows)
