"""Rating engine: precomputes ratings and serves predictions.

This module handles all the heavy computation at startup, then serves
results from memory for fast API responses.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from snooker_elo.data.loader import load_matches
from snooker_elo.ratings.base import RatingSystem
from snooker_elo.ratings.elo import EloRating
from snooker_elo.ratings.glicko2 import Glicko2Rating, Glicko2PlayerState


class RatingEngine:
    """Precomputed rating engine for the web API."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.matches: pd.DataFrame | None = None
        self.elo: EloRating | None = None
        self.glicko2: Glicko2Rating | None = None
        self._player_names: list[str] = []
        self._comparison_cache: dict | None = None

    def initialize(self):
        """Load data and compute all ratings. Called once at startup."""
        print("Loading match data...")
        start = time.time()
        self.matches = load_matches(self.data_path)
        print(f"  {len(self.matches)} matches loaded in {time.time()-start:.1f}s")

        print("Computing ELO ratings...")
        start = time.time()
        self.elo = EloRating(k_factor=9.77, divisor=327.15)
        self.elo.update(self.matches)
        print(f"  Done in {time.time()-start:.2f}s")

        print("Computing Glicko-2 ratings...")
        start = time.time()
        self.glicko2 = Glicko2Rating(tau=1.488, default_rating=1500)
        self.glicko2.update(self.matches)
        print(f"  Done in {time.time()-start:.2f}s")

        # Build sorted player name list
        all_names = set(self.elo.players.keys()) | set(self.glicko2.players.keys())
        self._player_names = sorted(all_names)
        print(f"  {len(self._player_names)} players indexed")

    def get_ratings(self, system: str = "elo", top: int = 50) -> list[dict]:
        """Get top-N player ratings for a given system."""
        if system == "elo":
            ratings = self.elo.get_ratings()
            players = self.elo.players
        else:
            ratings = self.glicko2.get_ratings()
            players = self.glicko2.players

        result = []
        for rank, (name, rating) in enumerate(ratings.head(top).items(), 1):
            state = players[name]
            entry = {
                "rank": rank,
                "name": name,
                "rating": round(rating),
                "matches_played": state.matches_played,
                "matches_won": state.matches_won,
                "win_rate": round(
                    state.matches_won / state.matches_played, 3
                ) if state.matches_played > 0 else 0,
            }
            if system == "glicko2" and isinstance(state, Glicko2PlayerState):
                entry["rd"] = round(state.rd, 1)
                entry["volatility"] = round(state.volatility, 4)
            result.append(entry)

        return result

    def get_player(self, name: str) -> dict | None:
        """Get detailed player profile with both rating systems."""
        elo_state = self.elo.players.get(name)
        g2_state = self.glicko2.players.get(name)

        if elo_state is None and g2_state is None:
            return None

        # Get recent matches
        player_matches = self.matches[
            (self.matches["player1"] == name) | (self.matches["player2"] == name)
        ].tail(20)

        recent = []
        for _, row in player_matches.iterrows():
            is_p1 = row["player1"] == name
            recent.append({
                "opponent": row["player2"] if is_p1 else row["player1"],
                "score": f"{row['score1']}-{row['score2']}" if is_p1
                    else f"{row['score2']}-{row['score1']}",
                "won": is_p1,  # player1 is always the winner in raw data
                "best_of": int(row["best_of"]),
                "year": int(row["year"]),
            })

        result = {"name": name, "recent_matches": recent}

        if elo_state:
            result["elo"] = {
                "rating": round(elo_state.rating),
                "matches_played": elo_state.matches_played,
                "matches_won": elo_state.matches_won,
                "win_rate": round(
                    elo_state.matches_won / elo_state.matches_played, 3
                ) if elo_state.matches_played > 0 else 0,
                "frames_played": elo_state.frames_played,
                "frames_won": elo_state.frames_won,
                "frame_win_rate": round(
                    elo_state.frames_won / elo_state.frames_played, 3
                ) if elo_state.frames_played > 0 else 0,
            }

        if g2_state:
            g2_data = {
                "rating": round(g2_state.rating),
                "matches_played": g2_state.matches_played,
                "matches_won": g2_state.matches_won,
                "win_rate": round(
                    g2_state.matches_won / g2_state.matches_played, 3
                ) if g2_state.matches_played > 0 else 0,
            }
            if isinstance(g2_state, Glicko2PlayerState):
                g2_data["rd"] = round(g2_state.rd, 1)
                g2_data["volatility"] = round(g2_state.volatility, 4)
            result["glicko2"] = g2_data

        return result

    def predict(self, player1: str, player2: str, best_of: int = 9) -> dict | None:
        """Predict match outcome between two players."""
        ep1 = self.elo.players.get(player1)
        ep2 = self.elo.players.get(player2)
        gp1 = self.glicko2.players.get(player1)
        gp2 = self.glicko2.players.get(player2)

        if ep1 is None and gp1 is None:
            return None
        if ep2 is None and gp2 is None:
            return None

        # ELO prediction
        elo_fp = self.elo.predict_frame_win_prob(player1, player2)
        elo_mp = self.elo.match_win_prob(elo_fp, best_of)

        # Glicko-2 prediction
        g2_fp = self.glicko2.predict_frame_win_prob(player1, player2)
        g2_mp = self.glicko2.match_win_prob(g2_fp, best_of)

        # Score probabilities for ELO
        win_cond = (best_of + 1) // 2
        from math import comb
        score_probs = []
        for s1 in range(win_cond, best_of + 1):
            s2 = s1 - win_cond
            total = s1 + s2
            # P(player1 wins with this score)
            p1_prob = comb(total - 1, s1 - 1) * elo_fp**(s1-1) * (1-elo_fp)**s2 * elo_fp
            score_probs.append({
                "score": f"{win_cond}-{s2}",
                "p1_prob": round(p1_prob, 4),
            })
        for s2 in range(win_cond, best_of + 1):
            s1 = s2 - win_cond
            total = s1 + s2
            p2_prob = comb(total - 1, s2 - 1) * (1-elo_fp)**(s2-1) * elo_fp**s1 * (1-elo_fp)
            score_probs.append({
                "score": f"{s1}-{win_cond}",
                "p1_prob": round(1 - p2_prob, 4),  # Store as complement
                "p2_prob": round(p2_prob, 4),
            })

        return {
            "player1": player1,
            "player2": player2,
            "best_of": best_of,
            "elo": {
                "player1_rating": round(ep1.rating) if ep1 else 1000,
                "player2_rating": round(ep2.rating) if ep2 else 1000,
                "frame_win_prob": round(elo_fp, 4),
                "match_win_prob": round(elo_mp, 4),
            },
            "glicko2": {
                "player1_rating": round(gp1.rating) if gp1 else 1500,
                "player2_rating": round(gp2.rating) if gp2 else 1500,
                "player1_rd": round(gp1.rd, 1) if isinstance(gp1, Glicko2PlayerState) else 350,
                "player2_rd": round(gp2.rd, 1) if isinstance(gp2, Glicko2PlayerState) else 350,
                "frame_win_prob": round(g2_fp, 4),
                "match_win_prob": round(g2_mp, 4),
            },
            "score_probabilities": score_probs,
        }

    def get_comparison(self) -> dict:
        """Get cached ELO vs Glicko-2 comparison."""
        if self._comparison_cache is not None:
            return self._comparison_cache

        # Simple comparison based on precomputed ratings
        elo_ratings = self.elo.get_ratings()
        g2_ratings = self.glicko2.get_ratings()

        # Top player overlap
        elo_top20 = set(elo_ratings.head(20).index)
        g2_top20 = set(g2_ratings.head(20).index)
        overlap = len(elo_top20 & g2_top20)

        # Find biggest disagreements
        disagreements = []
        for name in self._player_names:
            if name in self.elo.players and name in self.glicko2.players:
                ep = self.elo.players[name]
                gp = self.glicko2.players[name]
                if ep.matches_played >= 50:
                    elo_rank = list(elo_ratings.index).index(name) + 1 if name in elo_ratings.index else 9999
                    g2_rank = list(g2_ratings.index).index(name) + 1 if name in g2_ratings.index else 9999
                    diff = abs(elo_rank - g2_rank)
                    if diff > 10:
                        disagreements.append({
                            "name": name,
                            "elo_rank": elo_rank,
                            "glicko2_rank": g2_rank,
                            "rank_diff": diff,
                        })

        disagreements.sort(key=lambda x: x["rank_diff"], reverse=True)

        self._comparison_cache = {
            "top20_overlap": overlap,
            "total_players": len(self._player_names),
            "biggest_disagreements": disagreements[:10],
            "summary": {
                "elo": "Better accuracy and log-loss on match prediction",
                "glicko2": "Better calibration (ECE) and frame win MAE. "
                           "RD provides uncertainty estimates.",
            },
        }
        return self._comparison_cache

    def search_players(self, query: str) -> list[dict]:
        """Search players by name (case-insensitive prefix match)."""
        q = query.lower()
        results = []
        for name in self._player_names:
            if q in name.lower():
                ep = self.elo.players.get(name)
                results.append({
                    "name": name,
                    "elo_rating": round(ep.rating) if ep else None,
                    "matches_played": ep.matches_played if ep else 0,
                })
        results.sort(key=lambda x: -(x["matches_played"] or 0))
        return results

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {
            "total_matches": len(self.matches),
            "total_players": len(self._player_names),
            "year_range": {
                "min": int(self.matches["year"].min()),
                "max": int(self.matches["year"].max()),
            },
            "total_tournaments": len(self.matches["tournament_id"].unique()),
        }
