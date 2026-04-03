"""Rating engine: precomputes ratings and serves predictions.

This module handles all the heavy computation at startup, then serves
results from memory for fast API responses.
"""

from __future__ import annotations

import random
import time
from math import comb

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
        self._tournament_meta: dict[str, dict] = {}

    def initialize(self):
        """Load data and compute all ratings. Called once at startup."""
        print("Loading match data...")
        start = time.time()
        self.matches = load_matches(self.data_path)
        print(f"  {len(self.matches)} matches loaded in {time.time()-start:.1f}s")

        # Load tournament metadata
        from pathlib import Path
        tourn_path = Path(self.data_path).parent / "tournaments.csv"
        if tourn_path.exists():
            tdf = pd.read_csv(tourn_path, dtype={"tournament_id": str})
            for _, row in tdf.iterrows():
                self._tournament_meta[str(row["tournament_id"])] = {
                    "name": str(row.get("name", "")),
                    "status": str(row.get("status", "")),
                    "category": str(row.get("category", "")),
                    "city": str(row.get("city", "")),
                    "country": str(row.get("country", "")),
                }
            print(f"  {len(self._tournament_meta)} tournament metadata entries loaded")

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

    def get_rating_history(self, name: str) -> dict | None:
        """Get rating history for a player (ELO snapshots per tournament)."""
        ep = self.elo.players.get(name)
        if ep is None:
            return None

        # ELO history from rating_history field
        elo_history = []
        for entry in ep.rating_history:
            if len(entry) == 3:
                tid, rating, year = entry
                elo_history.append({"year": year, "rating": rating})

        # Sample to max ~200 points for charting
        if len(elo_history) > 200:
            step = len(elo_history) // 200
            elo_history = elo_history[::step]

        return {
            "name": name,
            "elo_history": elo_history,
            "current_elo": round(ep.rating),
        }

    def simulate_tournament(
        self, player_names: list[str], best_of: int = 9, n_sims: int = 10000
    ) -> dict:
        """Monte Carlo tournament bracket simulation.

        Simulates a single-elimination bracket n_sims times.
        Players are seeded by ELO rating (highest seed gets easiest draw).

        Returns win probability for each player at each round.
        """
        # Validate players and get frame win probabilities
        valid_players = []
        for name in player_names:
            if name in self.elo.players:
                valid_players.append(name)

        if len(valid_players) < 2:
            return {"error": "Need at least 2 valid players"}

        # Pad to power of 2 with byes (None)
        n = len(valid_players)
        bracket_size = 1
        while bracket_size < n:
            bracket_size *= 2

        # Seed by ELO rating
        valid_players.sort(key=lambda x: -self.elo.players[x].rating)
        bracket = list(valid_players) + [None] * (bracket_size - n)

        # Pre-compute all pairwise frame win probabilities
        probs = {}
        for i, p1 in enumerate(valid_players):
            for p2 in valid_players[i + 1:]:
                fp = self.elo.predict_frame_win_prob(p1, p2)
                mp = self.elo.match_win_prob(fp, best_of)
                probs[(p1, p2)] = mp
                probs[(p2, p1)] = 1.0 - mp

        # Count how many rounds
        n_rounds = 0
        temp = bracket_size
        while temp > 1:
            n_rounds += 1
            temp //= 2

        round_names = []
        if n_rounds >= 1:
            round_names = [f"Round {i+1}" for i in range(n_rounds)]
            if n_rounds >= 1:
                round_names[-1] = "Final"
            if n_rounds >= 2:
                round_names[-2] = "Semi-Final"
            if n_rounds >= 3:
                round_names[-3] = "Quarter-Final"

        # Simulate
        rng = random.Random(42)
        # Track: player -> [round1_advances, round2_advances, ..., wins]
        counts = {p: [0] * n_rounds for p in valid_players}

        for _ in range(n_sims):
            current = list(bracket)

            for round_idx in range(n_rounds):
                next_round = []
                for i in range(0, len(current), 2):
                    p1 = current[i]
                    p2 = current[i + 1] if i + 1 < len(current) else None

                    if p1 is None and p2 is None:
                        next_round.append(None)
                    elif p2 is None:
                        next_round.append(p1)
                        if p1:
                            counts[p1][round_idx] += 1
                    elif p1 is None:
                        next_round.append(p2)
                        counts[p2][round_idx] += 1
                    else:
                        prob_p1 = probs.get((p1, p2), 0.5)
                        if rng.random() < prob_p1:
                            next_round.append(p1)
                            counts[p1][round_idx] += 1
                        else:
                            next_round.append(p2)
                            counts[p2][round_idx] += 1

                current = next_round

        # Build results
        players_result = []
        for p in valid_players:
            entry = {
                "name": p,
                "elo_rating": round(self.elo.players[p].rating),
                "rounds": {},
            }
            for r_idx, r_name in enumerate(round_names):
                entry["rounds"][r_name] = round(counts[p][r_idx] / n_sims, 4)
            # Win probability is the last round
            entry["win_prob"] = round(counts[p][-1] / n_sims, 4) if n_rounds > 0 else 0
            players_result.append(entry)

        players_result.sort(key=lambda x: -x["win_prob"])

        return {
            "bracket_size": bracket_size,
            "n_players": len(valid_players),
            "n_simulations": n_sims,
            "best_of": best_of,
            "round_names": round_names,
            "players": players_result,
        }

    def get_recent_matches(self, limit: int = 50) -> dict:
        """Get most recent matches grouped by tournament."""
        # Get unique tournament IDs in reverse order (most recent first)
        all_tids = list(self.matches["tournament_id"].unique())
        recent_tids = all_tids[-limit:] if limit < len(all_tids) else all_tids
        recent_tids = list(reversed(recent_tids))

        tournaments = []
        for tid in recent_tids:
            tourn_df = self.matches[self.matches["tournament_id"] == tid]
            year = int(tourn_df["year"].iloc[0])
            n_matches = len(tourn_df)

            # Compute prediction accuracy for this tournament
            correct = 0
            matches_list = []
            for _, row in tourn_df.iterrows():
                p1, p2 = row["player1"], row["player2"]
                s1, s2, bo = int(row["score1"]), int(row["score2"]), int(row["best_of"])

                fp = self.elo.predict_frame_win_prob(p1, p2)
                mp = self.elo.match_win_prob(fp, bo)

                predicted_winner = p1 if mp > 0.5 else p2
                upset = predicted_winner != p1  # p1 is always actual winner
                if not upset:
                    correct += 1

                matches_list.append({
                    "player1": p1,
                    "player2": p2,
                    "score1": s1,
                    "score2": s2,
                    "best_of": bo,
                    "elo_win_prob": round(mp, 3),
                    "upset": upset,
                    "p1_elo": round(self.elo.players.get(
                        p1, type("", (), {"rating": 1000})
                    ).rating),
                    "p2_elo": round(self.elo.players.get(
                        p2, type("", (), {"rating": 1000})
                    ).rating),
                })

            meta = self._tournament_meta.get(str(tid), {})
            tournaments.append({
                "tournament_id": str(tid),
                "name": meta.get("name", f"Tournament #{tid}"),
                "status": meta.get("status", ""),
                "category": meta.get("category", ""),
                "city": meta.get("city", ""),
                "country": meta.get("country", ""),
                "year": year,
                "n_matches": n_matches,
                "accuracy": round(correct / n_matches, 3) if n_matches > 0 else 0,
                "upsets": n_matches - correct,
                "matches": matches_list,
            })

        return {"tournaments": tournaments, "total": len(tournaments)}

    def get_prime_times(self, min_matches: int = 200) -> dict:
        """Find peak rating and prime years for experienced players."""
        primes = []

        for name, state in self.elo.players.items():
            if state.matches_played < min_matches:
                continue
            if not state.rating_history or len(state.rating_history) < 3:
                continue

            # Find peak rating and when it occurred
            peak_rating = -1
            peak_year = 0
            first_year = state.rating_history[0][2] if len(state.rating_history[0]) >= 3 else 0
            last_year = state.rating_history[-1][2] if len(state.rating_history[-1]) >= 3 else 0

            for entry in state.rating_history:
                if len(entry) >= 3:
                    _, rating, year = entry
                    if rating > peak_rating:
                        peak_rating = rating
                        peak_year = year

            # Career span
            career_span = last_year - first_year if first_year > 0 and last_year > 0 else 0

            primes.append({
                "name": name,
                "current_rating": round(state.rating),
                "peak_rating": round(peak_rating),
                "peak_year": peak_year,
                "career_start": first_year,
                "career_end": last_year,
                "career_span": career_span,
                "matches_played": state.matches_played,
                "decline": round(peak_rating - state.rating),
                "win_rate": round(state.matches_won / state.matches_played, 3),
            })

        primes.sort(key=lambda x: -x["peak_rating"])
        return {"players": primes, "min_matches": min_matches}
