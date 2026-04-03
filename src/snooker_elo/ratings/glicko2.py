"""Glicko-2 rating system for snooker.

Implements Mark Glickman's Glicko-2 algorithm, adapted for snooker's
frame-based scoring. Each player has three parameters:
- Rating (μ): skill estimate
- Rating Deviation (RD/φ): uncertainty in the rating
- Volatility (σ): degree of expected fluctuation

Reference: http://www.glicko.net/glicko/glicko2.pdf
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass

import pandas as pd

from snooker_elo.ratings.base import PlayerState, RatingSystem

# Glicko-2 scaling constants
MU_SCALE = 173.7178  # Convert between Glicko-1 and Glicko-2 scale
EPSILON = 1e-6  # Convergence threshold for volatility iteration


@dataclass
class Glicko2PlayerState(PlayerState):
    """Extended player state with Glicko-2 specific fields."""

    rd: float = 350.0  # Rating Deviation (Glicko-1 scale)
    volatility: float = 0.06  # Performance consistency


class Glicko2Rating(RatingSystem):
    """Glicko-2 rating system adapted for professional snooker.

    Key adaptations from standard Glicko-2:
    - Frame-weighted outcomes: each match contributes proportionally to frames played.
    - Rating period = per-tournament: RDs inflate between tournaments.
    - Continuous outcomes: s = score1/(score1+score2) instead of binary win/loss.

    Parameters:
        tau: System volatility constraint (typically 0.3-1.2).
             Smaller = more stable ratings, larger = more reactive.
        initial_rd: Starting rating deviation for new players.
        initial_vol: Starting volatility for new players.
        default_rating: Starting rating (Glicko-1 scale).
    """

    def __init__(
        self,
        tau: float = 0.5,
        initial_rd: float = 350.0,
        initial_vol: float = 0.06,
        default_rating: float = 1500.0,
    ):
        super().__init__(default_rating=default_rating)
        self.tau = tau
        self.initial_rd = initial_rd
        self.initial_vol = initial_vol

    def _new_player_state(self) -> Glicko2PlayerState:
        return Glicko2PlayerState(
            rating=self.default_rating,
            rd=self.initial_rd,
            volatility=self.initial_vol,
        )

    def _ensure_player(self, name: str) -> Glicko2PlayerState:
        if name not in self.players:
            self.players[name] = self._new_player_state()
        return self.players[name]

    def _frame_win_prob(self, r1: float, r2: float) -> float:
        """Frame win probability using logistic function on Glicko-1 scale."""
        return 1.0 / (1.0 + math.exp(-(r1 - r2) / (400.0 / math.log(10))))

    @staticmethod
    def _g(phi: float) -> float:
        """g(φ) function: reduces impact of opponent's rating based on their uncertainty."""
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)

    @staticmethod
    def _E(mu: float, mu_j: float, phi_j: float) -> float:
        """Expected score against opponent j on Glicko-2 scale."""
        return 1.0 / (1.0 + math.exp(-Glicko2Rating._g(phi_j) * (mu - mu_j)))

    def _compute_variance(
        self, mu: float, opponents: list[tuple[float, float, float]]
    ) -> float:
        """Compute estimated variance of player's rating (Step 3 of algorithm).

        Args:
            mu: Player's rating on Glicko-2 scale.
            opponents: List of (mu_j, phi_j, weight_j) for each opponent interaction.

        Returns:
            Variance v.
        """
        v_inv = 0.0
        for mu_j, phi_j, _ in opponents:
            g_j = self._g(phi_j)
            e_j = self._E(mu, mu_j, phi_j)
            v_inv += g_j**2 * e_j * (1.0 - e_j)
        return 1.0 / v_inv if v_inv > 0 else 1e10

    def _compute_delta(
        self, mu: float, opponents: list[tuple[float, float, float, float]]
    ) -> float:
        """Compute the quantity Δ (Step 4: weighted performance vs expectation).

        Args:
            mu: Player's rating on Glicko-2 scale.
            opponents: List of (mu_j, phi_j, s_j, weight_j) for each interaction.

        Returns:
            Delta value.
        """
        total = 0.0
        for mu_j, phi_j, s_j, _ in opponents:
            g_j = self._g(phi_j)
            e_j = self._E(mu, mu_j, phi_j)
            total += g_j * (s_j - e_j)
        return total

    def _new_volatility(
        self, sigma: float, phi: float, v: float, delta: float
    ) -> float:
        """Compute new volatility using the Illinois algorithm (Step 5).

        This solves: f(σ') = 0 where f is defined in Glickman's paper.
        """
        tau = self.tau
        a = math.log(sigma**2)

        def f(x: float) -> float:
            ex = math.exp(x)
            num = ex * (delta**2 - phi**2 - v - ex)
            denom = 2.0 * (phi**2 + v + ex) ** 2
            return num / denom - (x - a) / tau**2

        # Initialize bounds
        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau

        # Illinois algorithm
        f_A = f(A)
        f_B = f(B)
        for _ in range(100):  # Max iterations
            if abs(B - A) < EPSILON:
                break
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)
            if f_C * f_B <= 0:
                A = B
                f_A = f_B
            else:
                f_A /= 2.0
            B = C
            f_B = f_C

        return math.exp(A / 2.0)

    def _update_player(
        self,
        state: Glicko2PlayerState,
        opponents: list[tuple[float, float, float, float]],
    ) -> None:
        """Update a single player's rating after a rating period.

        Args:
            state: The player's current state (modified in place).
            opponents: List of (mu_j, phi_j, s_j, weight_j) — one per match.
                       s_j is the frame win proportion.
                       weight_j is total frames (used for frame-weighted update).
        """
        if not opponents:
            # No games played: only increase RD (Step 6a)
            phi = state.rd / MU_SCALE
            phi_star = math.sqrt(phi**2 + state.volatility**2)
            state.rd = phi_star * MU_SCALE
            return

        mu = (state.rating - self.default_rating) / MU_SCALE
        phi = state.rd / MU_SCALE

        # Build weighted opponent list for variance calculation
        opp_for_var = [(mu_j, phi_j, w) for mu_j, phi_j, _, w in opponents]
        v = self._compute_variance(mu, opp_for_var)
        delta_raw = self._compute_delta(mu, opponents)
        delta = v * delta_raw

        # Step 5: New volatility
        sigma_new = self._new_volatility(state.volatility, phi, v, delta)

        # Step 6: Update phi and mu
        phi_star = math.sqrt(phi**2 + sigma_new**2)
        phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        mu_new = mu + phi_new**2 * delta_raw

        # Convert back to Glicko-1 scale
        state.rating = mu_new * MU_SCALE + self.default_rating
        state.rd = phi_new * MU_SCALE
        state.volatility = sigma_new

    def update(self, matches: pd.DataFrame) -> None:
        """Process all matches grouped by tournament (= rating period).

        For each tournament:
        1. Collect all match results per player.
        2. Update each player's rating, RD, and volatility.
        3. Update career/time-window statistics.

        Args:
            matches: DataFrame with columns [player1, player2, score1, score2,
                     best_of, tournament_id, date, year].
        """
        current_year = int(matches["year"].max())
        players = self.players

        # Pre-register all players
        all_names = pd.concat([matches["player1"], matches["player2"]]).unique()
        for name in all_names:
            if name not in players:
                players[name] = self._new_player_state()

        # Group by tournament (= rating period)
        for _, tourn_matches in matches.groupby("tournament_id", sort=False):
            # Snapshot ratings at start of tournament for opponent lookups
            ratings_snapshot: dict[str, tuple[float, float]] = {}
            for name, state in players.items():
                mu = (state.rating - self.default_rating) / MU_SCALE
                phi = state.rd / MU_SCALE
                ratings_snapshot[name] = (mu, phi)

            # Collect opponents per player for this rating period
            player_opponents: dict[str, list[tuple[float, float, float, float]]] = {}

            for row in tourn_matches.itertuples(index=False):
                p1_name = row.player1
                p2_name = row.player2
                score1 = int(row.score1)
                score2 = int(row.score2)
                year = int(row.year)
                total = score1 + score2

                # Frame win proportions
                s1 = score1 / total
                s2 = score2 / total

                mu1, phi1 = ratings_snapshot[p1_name]
                mu2, phi2 = ratings_snapshot[p2_name]

                # Weight by total frames
                weight = float(total)

                player_opponents.setdefault(p1_name, []).append((mu2, phi2, s1, weight))
                player_opponents.setdefault(p2_name, []).append((mu1, phi1, s2, weight))

                # Update career stats (same as ELO)
                p1 = players[p1_name]
                p2 = players[p2_name]

                p1.matches_played += 1
                p2.matches_played += 1
                p1.matches_won += 1

                p1.frames_played += total
                p2.frames_played += total
                p1.frames_won += score1
                p2.frames_won += score2

                if current_year - year < 3:
                    p1.matches_played_3y += 1
                    p2.matches_played_3y += 1
                    p1.matches_won_3y += 1
                    p1.frames_played_3y += total
                    p2.frames_played_3y += total
                    p1.frames_won_3y += score1
                    p2.frames_won_3y += score2

                if current_year - year < 1:
                    p1.matches_played_1y += 1
                    p2.matches_played_1y += 1
                    p1.matches_won_1y += 1
                    p1.frames_played_1y += total
                    p2.frames_played_1y += total
                    p1.frames_won_1y += score1
                    p2.frames_won_1y += score2

            # Update ratings for all active players in this tournament
            for name, opps in player_opponents.items():
                self._update_player(players[name], opps)

            # Inflate RD for inactive players (didn't play in this tournament)
            for name, state in players.items():
                if name not in player_opponents:
                    phi = state.rd / MU_SCALE
                    phi_star = math.sqrt(phi**2 + state.volatility**2)
                    state.rd = min(phi_star * MU_SCALE, self.initial_rd)

    def get_player_stats(self) -> pd.DataFrame:
        """Return player stats including Glicko-2 specific fields (RD, volatility)."""
        rows = {}
        for name, s in self.players.items():
            rows[name] = {
                "rating": round(s.rating),
                "rd": round(s.rd, 1),
                "volatility": round(s.volatility, 4),
                "matches_played": s.matches_played,
                "matches_won": s.matches_won,
                "matches_win_rate": s.matches_won / s.matches_played if s.matches_played else 0.0,
                "frames_played": s.frames_played,
                "frames_won": s.frames_won,
                "frames_win_rate": s.frames_won / s.frames_played if s.frames_played else 0.0,
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def snapshot(self) -> dict[str, Glicko2PlayerState]:
        """Return a shallow copy of all player states."""
        return {name: copy(state) for name, state in self.players.items()}

    def generate_match_features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Generate features including Glicko-2 specific columns (RD)."""
        players = self.players

        rows = []
        for row in matches.itertuples(index=False):
            p1_name = row.player1
            p2_name = row.player2
            best_of = int(row.best_of)

            p1 = players.get(p1_name, self._new_player_state())
            p2 = players.get(p2_name, self._new_player_state())

            fp = self._frame_win_prob(p1.rating, p2.rating)
            mp = self.match_win_prob(fp, best_of)

            rows.append(
                {
                    "player1": p1_name,
                    "player2": p2_name,
                    "best_of": best_of,
                    "player1_glicko2": round(p1.rating),
                    "player2_glicko2": round(p2.rating),
                    "player1_rd": round(p1.rd, 1),
                    "player2_rd": round(p2.rd, 1),
                    "glicko2_match_win_rate": mp,
                    "glicko2_frame_win_rate": fp,
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
