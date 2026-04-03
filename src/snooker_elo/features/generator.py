"""Incremental feature generation pipeline.

Replaces the legacy O(T²·M) data_generation.ipynb with an O(T·M) single-pass
approach. For each tournament, snapshots current player states, generates features,
then updates the rating system with that tournament's results.

Feature design: focuses on non-redundant, informative features:
- Rating predictions (ELO + Glicko-2 frame/match probabilities)
- Uncertainty (Glicko-2 RD)
- Win rate differences (career, 1y, 3y) instead of raw counts
- Head-to-head record
- Inactivity and momentum
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd

from snooker_elo.ratings.base import PlayerState
from snooker_elo.ratings.elo import EloRating
from snooker_elo.ratings.glicko2 import Glicko2PlayerState, Glicko2Rating


def generate_features(
    matches: pd.DataFrame,
    elo: EloRating | None = None,
    glicko2: Glicko2Rating | None = None,
    start_tournament_idx: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate ML features incrementally over tournaments.

    For each tournament N (starting from start_tournament_idx):
    1. Snapshot current player states from both rating systems.
    2. Generate features for each match in tournament N using the snapshot.
    3. Update both rating systems with tournament N's results.

    This is O(T·M) instead of the legacy O(T²·M) approach.

    Args:
        matches: Full match DataFrame, ordered chronologically.
        elo: EloRating instance (created with defaults if None).
        glicko2: Glicko2Rating instance (created with defaults if None).
        start_tournament_idx: Index into tournament list to start generating
            features from. Earlier tournaments are used only for warm-up.
        seed: Random seed for player swap randomization.

    Returns:
        DataFrame with features for all matches from start_tournament_idx onwards,
        with 50% of rows randomly swapped (player1 ↔ player2) to remove bias.
    """
    if elo is None:
        elo = EloRating(k_factor=8, divisor=400, default_rating=1000)
    if glicko2 is None:
        glicko2 = Glicko2Rating(tau=0.5, default_rating=1500)

    tournament_ids = list(matches["tournament_id"].unique())
    all_feature_rows = []

    # Track head-to-head records incrementally: (p1, p2) -> [p1_wins, p2_wins]
    h2h: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    match_counter = 0

    for t_idx, tid in enumerate(tournament_ids):
        tourn_matches = matches[matches["tournament_id"] == tid]

        if t_idx >= start_tournament_idx:
            rows = _generate_tournament_features(
                tourn_matches, elo, glicko2, h2h, match_counter,
            )
            all_feature_rows.extend(rows)

        # Update h2h records and match counter
        for row in tourn_matches.itertuples(index=False):
            p1, p2 = row.player1, row.player2
            key = tuple(sorted([p1, p2]))
            if p1 == key[0]:
                h2h[key][0] += 1
            else:
                h2h[key][1] += 1
            match_counter += 1

        # Update both rating systems with this tournament
        elo.update(tourn_matches)
        glicko2.update(tourn_matches)

    if not all_feature_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_feature_rows)

    # Randomly swap 50% of matches to remove player1-always-wins bias
    rng = random.Random(seed)
    swap_mask = [rng.random() < 0.5 for _ in range(len(df))]
    df = _apply_swap(df, swap_mask)

    return df


def _safe_ratio(num: int, denom: int) -> float:
    """Compute ratio, returning 0.0 if denominator is 0."""
    return num / denom if denom > 0 else 0.0


def _generate_tournament_features(
    tourn_matches: pd.DataFrame,
    elo: EloRating,
    glicko2: Glicko2Rating,
    h2h: dict[tuple[str, str], list[int]],
    match_counter: int,
) -> list[dict]:
    """Generate feature rows for one tournament's matches.

    Features are designed to be non-redundant:
    - Predictions (probabilities) instead of raw ratings
    - Win rate DIFFERENCES instead of raw counts for both players
    - Head-to-head, momentum, and inactivity as new signals
    """
    elo_players = elo.players
    g2_players = glicko2.players
    elo_default = elo.default_rating
    g2_default_state = glicko2._new_player_state()

    rows = []
    for row in tourn_matches.itertuples(index=False):
        p1_name = row.player1
        p2_name = row.player2
        score1 = int(row.score1)
        score2 = int(row.score2)
        best_of = int(row.best_of)

        # ELO state
        ep1 = elo_players.get(p1_name, PlayerState(rating=elo_default))
        ep2 = elo_players.get(p2_name, PlayerState(rating=elo_default))

        elo_fp = elo._frame_win_prob(ep1.rating, ep2.rating)
        elo_mp = elo.match_win_prob(elo_fp, best_of)

        # Glicko-2 state
        gp1 = g2_players.get(p1_name, g2_default_state)
        gp2 = g2_players.get(p2_name, g2_default_state)

        g2_fp = glicko2._frame_win_prob(gp1.rating, gp2.rating)
        g2_mp = glicko2.match_win_prob(g2_fp, best_of)

        # Win rate differences (non-redundant: difference instead of raw counts)
        p1_match_wr = _safe_ratio(ep1.matches_won, ep1.matches_played)
        p2_match_wr = _safe_ratio(ep2.matches_won, ep2.matches_played)
        p1_frame_wr = _safe_ratio(ep1.frames_won, ep1.frames_played)
        p2_frame_wr = _safe_ratio(ep2.frames_won, ep2.frames_played)
        p1_1y_wr = _safe_ratio(ep1.frames_won_1y, ep1.frames_played_1y)
        p2_1y_wr = _safe_ratio(ep2.frames_won_1y, ep2.frames_played_1y)
        p1_3y_wr = _safe_ratio(ep1.frames_won_3y, ep1.frames_played_3y)
        p2_3y_wr = _safe_ratio(ep2.frames_won_3y, ep2.frames_played_3y)

        # Head-to-head
        h2h_key = tuple(sorted([p1_name, p2_name]))
        h2h_record = h2h.get(h2h_key, [0, 0])
        if p1_name == h2h_key[0]:
            p1_h2h_wins, p2_h2h_wins = h2h_record
        else:
            p2_h2h_wins, p1_h2h_wins = h2h_record
        h2h_total = p1_h2h_wins + p2_h2h_wins
        h2h_advantage = _safe_ratio(p1_h2h_wins, h2h_total) - 0.5 if h2h_total > 0 else 0.0

        # Momentum: rating change over recent matches
        p1_momentum = ep1.rating - ep1.prev_rating if ep1.prev_rating > 0 else 0.0
        p2_momentum = ep2.rating - ep2.prev_rating if ep2.prev_rating > 0 else 0.0

        # Inactivity: matches since last game (capped at 1000)
        p1_inact = ep1.last_match_idx
        p1_inactivity = min(match_counter - p1_inact, 1000) if p1_inact > 0 else 1000
        p2_inact = ep2.last_match_idx
        p2_inactivity = min(match_counter - p2_inact, 1000) if p2_inact > 0 else 1000

        features = {
            # Match info
            "player1": p1_name,
            "player2": p2_name,
            "best_of": best_of,
            # === Rating predictions (4 features) ===
            "elo_frame_win_rate": elo_fp,
            "elo_match_win_rate": elo_mp,
            "glicko2_frame_win_rate": g2_fp,
            "glicko2_match_win_rate": g2_mp,
            # === Uncertainty (2 features) ===
            "player1_rd": round(gp1.rd, 1) if isinstance(gp1, Glicko2PlayerState) else 350.0,
            "player2_rd": round(gp2.rd, 1) if isinstance(gp2, Glicko2PlayerState) else 350.0,
            # === Win rate differences (4 features, not raw counts) ===
            "match_wr_diff": p1_match_wr - p2_match_wr,
            "frame_wr_diff": p1_frame_wr - p2_frame_wr,
            "form_1y_diff": p1_1y_wr - p2_1y_wr,
            "form_3y_diff": p1_3y_wr - p2_3y_wr,
            # === Experience (1 feature) ===
            "experience_diff": ep1.matches_played - ep2.matches_played,
            # === NEW: Head-to-head (2 features) ===
            "h2h_advantage": h2h_advantage,
            "h2h_matches": h2h_total,
            # === NEW: Momentum (1 feature) ===
            "momentum_diff": p1_momentum - p2_momentum,
            # === NEW: Inactivity (1 feature) ===
            "inactivity_diff": p2_inactivity - p1_inactivity,  # Positive = p1 more active
            # === Raw ratings (kept for the web app / feature generation) ===
            "player1_elo": round(ep1.rating),
            "player2_elo": round(ep2.rating),
            "player1_glicko2": round(gp1.rating),
            "player2_glicko2": round(gp2.rating),
            # === Raw stats (kept for backward compat, but NOT used in ML) ===
            "p1_matches_played": ep1.matches_played,
            "p1_matches_won": ep1.matches_won,
            "p1_frames_played": ep1.frames_played,
            "p1_frames_won": ep1.frames_won,
            "p2_matches_played": ep2.matches_played,
            "p2_matches_won": ep2.matches_won,
            "p2_frames_played": ep2.frames_played,
            "p2_frames_won": ep2.frames_won,
            "p1_frames_played_1_year": ep1.frames_played_1y,
            "p1_frames_won_1_year": ep1.frames_won_1y,
            "p1_frames_played_3_years": ep1.frames_played_3y,
            "p1_frames_won_3_years": ep1.frames_won_3y,
            "p2_frames_played_1_year": ep2.frames_played_1y,
            "p2_frames_won_1_year": ep2.frames_won_1y,
            "p2_frames_played_3_years": ep2.frames_played_3y,
            "p2_frames_won_3_years": ep2.frames_won_3y,
            # Target variables
            "score1": score1,
            "score2": score2,
            "match_result": 0,
            "win_percentage": score1 / (score1 + score2),
            # Metadata
            "tournament_id": str(row.tournament_id),
            "date": str(getattr(row, "date", "")),
        }
        rows.append(features)

    return rows


def _apply_swap(df: pd.DataFrame, swap_mask: list[bool]) -> pd.DataFrame:
    """Randomly swap player1/player2 for rows where swap_mask is True.

    This removes the bias that player1 is always the winner in the raw data.
    """
    df = df.copy()

    # Columns to swap between p1 and p2
    swap_pairs = [
        ("player1", "player2"),
        ("player1_elo", "player2_elo"),
        ("player1_glicko2", "player2_glicko2"),
        ("player1_rd", "player2_rd"),
        ("p1_matches_played", "p2_matches_played"),
        ("p1_matches_won", "p2_matches_won"),
        ("p1_frames_played", "p2_frames_played"),
        ("p1_frames_won", "p2_frames_won"),
        ("p1_frames_played_1_year", "p2_frames_played_1_year"),
        ("p1_frames_won_1_year", "p2_frames_won_1_year"),
        ("p1_frames_played_3_years", "p2_frames_played_3_years"),
        ("p1_frames_won_3_years", "p2_frames_won_3_years"),
        ("score1", "score2"),
    ]

    # Signed features that need to be negated on swap
    negate_cols = [
        "match_wr_diff", "frame_wr_diff", "form_1y_diff", "form_3y_diff",
        "experience_diff", "h2h_advantage", "momentum_diff", "inactivity_diff",
    ]

    mask = np.array(swap_mask)
    for col_a, col_b in swap_pairs:
        a_vals = df[col_a].values.copy()
        b_vals = df[col_b].values.copy()
        df.loc[mask, col_a] = b_vals[mask]
        df.loc[mask, col_b] = a_vals[mask]

    # Flip derived columns
    df.loc[mask, "match_result"] = 1  # Swapped: player2 (original winner) is now player2
    df.loc[mask, "win_percentage"] = 1.0 - df.loc[mask, "win_percentage"]
    df.loc[mask, "elo_frame_win_rate"] = 1.0 - df.loc[mask, "elo_frame_win_rate"]
    df.loc[mask, "elo_match_win_rate"] = 1.0 - df.loc[mask, "elo_match_win_rate"]
    df.loc[mask, "glicko2_frame_win_rate"] = 1.0 - df.loc[mask, "glicko2_frame_win_rate"]
    df.loc[mask, "glicko2_match_win_rate"] = 1.0 - df.loc[mask, "glicko2_match_win_rate"]

    # Negate signed difference features
    for col in negate_cols:
        if col in df.columns:
            df.loc[mask, col] = -df.loc[mask, col]

    return df.reset_index(drop=True)
