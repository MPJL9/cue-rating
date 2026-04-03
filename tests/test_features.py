"""Tests for incremental feature generation pipeline."""

import pandas as pd
import pytest

from snooker_elo.features.generator import _apply_swap, generate_features


def _make_matches(*rows):
    data = []
    for r in rows:
        data.append({
            "player1": r[0], "player2": r[1], "score1": r[2], "score2": r[3],
            "best_of": r[4], "tournament_id": str(r[5]), "date": "", "year": r[6],
        })
    return pd.DataFrame(data)


class TestGenerateFeatures:
    def test_basic_output_shape(self):
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2024),
            ("C", "D", 6, 2, 9, "T1", 2024),
            ("A", "C", 5, 4, 9, "T2", 2024),
        )
        # Warm up on T1, generate features for T2
        df = generate_features(matches, start_tournament_idx=1)
        assert len(df) == 1  # Only T2 matches

    def test_all_tournaments(self):
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2024),
            ("A", "C", 5, 4, 9, "T2", 2024),
        )
        df = generate_features(matches, start_tournament_idx=0)
        assert len(df) == 2  # Both tournaments

    def test_has_both_elo_and_glicko2(self):
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2024),
            ("A", "B", 5, 2, 9, "T2", 2024),
        )
        df = generate_features(matches, start_tournament_idx=1)
        assert "player1_elo" in df.columns
        assert "player1_glicko2" in df.columns
        assert "player1_rd" in df.columns
        assert "elo_frame_win_rate" in df.columns
        assert "glicko2_frame_win_rate" in df.columns

    def test_no_data_leakage(self):
        """Features for tournament N should NOT use tournament N's results."""
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2024),
            ("A", "B", 9, 0, 17, "T2", 2024),
        )
        # Features for T1 generated before any updates
        df = generate_features(matches, start_tournament_idx=0)
        t1_row = df[df["tournament_id"] == "T1"].iloc[0]
        # Both players should have 0 matches played at the start
        assert t1_row["p1_matches_played"] == 0
        assert t1_row["p2_matches_played"] == 0

    def test_swap_randomization(self):
        """~50% of rows should be swapped."""
        matches = _make_matches(
            *[("A", "B", 5, 3, 9, "T1", 2024) for _ in range(100)]
        )
        df = generate_features(matches, start_tournament_idx=0, seed=42)
        # match_result should be ~50% zeros and ~50% ones
        swap_rate = df["match_result"].mean()
        assert 0.3 < swap_rate < 0.7


class TestApplySwap:
    def test_swap_flips_players(self):
        df = pd.DataFrame({
            "player1": ["A"], "player2": ["B"],
            "player1_elo": [1100], "player2_elo": [900],
            "player1_glicko2": [1600], "player2_glicko2": [1400],
            "player1_rd": [50.0], "player2_rd": [100.0],
            "p1_matches_played": [10], "p2_matches_played": [5],
            "p1_matches_won": [7], "p2_matches_won": [3],
            "p1_frames_played": [80], "p2_frames_played": [40],
            "p1_frames_won": [50], "p2_frames_won": [20],
            "p1_frames_played_1_year": [30], "p2_frames_played_1_year": [15],
            "p1_frames_won_1_year": [20], "p2_frames_won_1_year": [8],
            "p1_frames_played_3_years": [60], "p2_frames_played_3_years": [30],
            "p1_frames_won_3_years": [40], "p2_frames_won_3_years": [15],
            "score1": [5], "score2": [3],
            "match_result": [0], "win_percentage": [0.625],
            "elo_frame_win_rate": [0.6], "elo_match_win_rate": [0.7],
            "glicko2_frame_win_rate": [0.65], "glicko2_match_win_rate": [0.75],
            "tournament_id": ["T1"], "date": [""],
        })
        result = _apply_swap(df, [True])

        assert result.iloc[0]["player1"] == "B"
        assert result.iloc[0]["player2"] == "A"
        assert result.iloc[0]["player1_elo"] == 900
        assert result.iloc[0]["player2_elo"] == 1100
        assert result.iloc[0]["match_result"] == 1
        assert result.iloc[0]["win_percentage"] == pytest.approx(0.375)
        assert result.iloc[0]["elo_frame_win_rate"] == pytest.approx(0.4)


class TestOnRealData:
    def test_runs_on_real_data(self):
        """Feature generation should work on actual match data."""
        from snooker_elo.data.loader import load_matches
        matches = load_matches().head(2000)  # ~15 tournaments

        tournament_ids = list(matches["tournament_id"].unique())
        n_tourns = len(tournament_ids)

        # Warm up on first half, generate for second half
        start = n_tourns // 2
        df = generate_features(matches, start_tournament_idx=start)

        assert len(df) > 0
        assert "player1_elo" in df.columns
        assert "player1_glicko2" in df.columns
        assert df["match_result"].isin([0, 1]).all()
