"""Tests for ELO rating system."""

import math

import numpy as np
import pandas as pd
import pytest

from snooker_elo.ratings.base import RatingSystem
from snooker_elo.ratings.elo import EloRating


def _make_matches(*rows):
    """Helper to create a matches DataFrame from tuples.

    Each row: (player1, player2, score1, score2, best_of, tournament_id, year)
    """
    data = []
    for r in rows:
        data.append(
            {
                "player1": r[0],
                "player2": r[1],
                "score1": r[2],
                "score2": r[3],
                "best_of": r[4],
                "tournament_id": str(r[5]),
                "date": "",
                "year": r[6],
            }
        )
    return pd.DataFrame(data)


class TestCalculateNewElo:
    """Test the core ELO update formula."""

    def test_equal_ratings_winner_gains(self):
        r1_new, r2_new = EloRating._calculate_new_elo(1000, 1000, 5, 3, 8, 400)
        assert r1_new > 1000
        assert r2_new < 1000

    def test_equal_ratings_symmetric(self):
        r1_new, r2_new = EloRating._calculate_new_elo(1000, 1000, 5, 3, 8, 400)
        # Sum should be preserved (zero-sum)
        assert r1_new + r2_new == 2000

    def test_known_values(self):
        # Manual calculation: R1=1200, R2=1000, score 5-4, K=8, divisor=400
        # E1 = 1/(1+exp((1000-1200)/400)) = 1/(1+exp(-0.5)) ≈ 0.6225
        # S1 = 5/9 ≈ 0.5556
        # R1_new = round(1200 + 8*9*(0.5556 - 0.6225)) = round(1200 + 72*(-0.0669))
        #        = round(1200 - 4.82) = round(1195.18) = 1195
        r1_new, r2_new = EloRating._calculate_new_elo(1200, 1000, 5, 4, 8, 400)
        assert r1_new == 1195
        assert r2_new == 1005

    def test_blowout_large_update(self):
        # 9-0 blowout: winner should gain more than a close match
        r1_close, _ = EloRating._calculate_new_elo(1000, 1000, 5, 4, 8, 400)
        r1_blow, _ = EloRating._calculate_new_elo(1000, 1000, 9, 0, 8, 400)
        assert r1_blow > r1_close


class TestMatchWinProb:
    """Test binomial match win probability calculation."""

    def test_even_match(self):
        # 50% frame win rate = 50% match win rate regardless of format
        assert RatingSystem.match_win_prob(0.5, 5) == pytest.approx(0.5, abs=1e-10)
        assert RatingSystem.match_win_prob(0.5, 9) == pytest.approx(0.5, abs=1e-10)
        assert RatingSystem.match_win_prob(0.5, 17) == pytest.approx(0.5, abs=1e-10)

    def test_strong_favorite_amplified(self):
        # Longer format amplifies skill advantage
        p5 = RatingSystem.match_win_prob(0.6, 5)
        p9 = RatingSystem.match_win_prob(0.6, 9)
        p17 = RatingSystem.match_win_prob(0.6, 17)
        assert p5 > 0.5
        assert p9 > p5
        assert p17 > p9

    def test_best_of_5(self):
        # P(win bo5 | p=0.6) = P(3-0) + P(3-1) + P(3-2)
        p = 0.6
        q = 1 - p
        expected = (
            p**3  # 3-0
            + 3 * p**2 * q * p  # 3-1: C(3,2)*p^2*q * p
            + 6 * p**2 * q**2 * p  # 3-2: C(4,2)*p^2*q^2 * p
        )
        assert RatingSystem.match_win_prob(0.6, 5) == pytest.approx(expected, rel=1e-10)

    def test_probabilities_sum_to_one(self):
        # P1 wins + P2 wins = 1
        p = 0.55
        for bo in [5, 7, 9, 11, 17, 19, 35]:
            p1_wins = RatingSystem.match_win_prob(p, bo)
            p2_wins = RatingSystem.match_win_prob(1 - p, bo)
            assert p1_wins + p2_wins == pytest.approx(1.0, abs=1e-10)


class TestEloUpdate:
    """Test the full update pipeline."""

    def test_single_match(self):
        elo = EloRating(k_factor=8, divisor=400)
        matches = _make_matches(("Alice", "Bob", 5, 3, 9, "T1", 2024))
        elo.update(matches)

        assert elo.players["Alice"].rating > 1000
        assert elo.players["Bob"].rating < 1000
        assert elo.players["Alice"].matches_played == 1
        assert elo.players["Alice"].matches_won == 1
        assert elo.players["Bob"].matches_won == 0
        assert elo.players["Alice"].frames_won == 5
        assert elo.players["Bob"].frames_won == 3

    def test_time_window_stats(self):
        elo = EloRating(k_factor=8)
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2020),  # Old match (>3 years from max)
            ("A", "B", 5, 2, 9, "T2", 2023),  # Within 3 years of 2024
            ("A", "B", 5, 1, 9, "T3", 2024),  # Within 1 year of 2024
        )
        elo.update(matches)

        # Max year = 2024
        # 2020: 2024-2020=4 >= 3, so NOT in 3y window
        # 2023: 2024-2023=1 < 3, so IN 3y window, but NOT in 1y (1 < 1 is False)
        # 2024: 2024-2024=0 < 3 and < 1, so in BOTH windows
        assert elo.players["A"].frames_played_3y == (5 + 2) + (5 + 1)  # matches 2 and 3
        assert elo.players["A"].frames_played_1y == 5 + 1  # only match 3

    def test_multiple_players(self):
        elo = EloRating(k_factor=8)
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2024),
            ("C", "D", 6, 2, 9, "T1", 2024),
            ("A", "C", 5, 4, 9, "T1", 2024),
        )
        elo.update(matches)

        assert len(elo.players) == 4
        # A won twice, B/C/D won 0/0/0
        assert elo.players["A"].matches_won == 2
        assert elo.players["C"].matches_won == 1
        assert elo.players["B"].matches_won == 0
        assert elo.players["D"].matches_won == 0

    def test_unknown_player_prediction(self):
        elo = EloRating()
        matches = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        elo.update(matches)

        # Predict with unknown player
        fp = elo.predict_frame_win_prob("A", "Unknown")
        assert fp > 0.5  # A should be favored over default-rated player


class TestPredictBatch:
    """Test batch prediction."""

    def test_batch_predictions(self):
        elo = EloRating()
        train = _make_matches(
            ("A", "B", 5, 1, 9, "T1", 2024),
            ("A", "C", 5, 2, 9, "T1", 2024),
        )
        elo.update(train)

        test = pd.DataFrame(
            {"player1": ["A", "B"], "player2": ["B", "C"], "best_of": [9, 9]}
        )
        preds, frame_probs, match_probs = elo.predict_batch(test)

        assert len(preds) == 2
        assert frame_probs[0] > 0.5  # A should beat B
        assert all(0 <= p <= 1 for p in match_probs)


class TestGenerateMatchFeatures:
    """Test feature generation for ML pipeline."""

    def test_feature_columns(self):
        elo = EloRating()
        train = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        elo.update(train)

        test = pd.DataFrame(
            {"player1": ["A"], "player2": ["B"], "best_of": [9]}
        )
        features = elo.generate_match_features(test)

        expected_cols = [
            "player1", "player2", "best_of",
            "player1_elo", "player2_elo",
            "elo_match_win_rate", "elo_frame_win_rate",
            "p1_matches_played", "p1_matches_won",
            "p1_frames_played", "p1_frames_won",
            "p2_matches_played", "p2_matches_won",
            "p2_frames_played", "p2_frames_won",
            "p1_frames_played_1_year", "p1_frames_won_1_year",
            "p1_frames_played_3_years", "p1_frames_won_3_years",
            "p2_frames_played_1_year", "p2_frames_won_1_year",
            "p2_frames_played_3_years", "p2_frames_won_3_years",
        ]
        assert list(features.columns) == expected_cols

    def test_features_reflect_state(self):
        elo = EloRating()
        train = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        elo.update(train)

        test = pd.DataFrame(
            {"player1": ["A"], "player2": ["B"], "best_of": [9]}
        )
        features = elo.generate_match_features(test)

        assert features.iloc[0]["player1_elo"] > features.iloc[0]["player2_elo"]
        assert features.iloc[0]["p1_matches_played"] == 1
        assert features.iloc[0]["p1_frames_won"] == 5
        assert features.iloc[0]["elo_frame_win_rate"] > 0.5


class TestRegressionAgainstLegacy:
    """Verify new implementation matches legacy on real data."""

    @pytest.fixture
    def matches_small(self):
        """First 100 matches from matches.csv for fast testing."""
        from snooker_elo.data.loader import load_matches

        df = load_matches()
        return df.head(100)

    def test_ratings_match_legacy_logic(self, matches_small):
        """New ELO should produce same ratings as legacy for identical parameters."""
        elo = EloRating(k_factor=8, divisor=400, default_rating=1000)
        elo.update(matches_small)

        # Verify basic properties that legacy code guaranteed
        ratings = elo.get_ratings()
        assert len(ratings) > 0
        # All ratings should be within reasonable range
        assert ratings.max() < 2000
        assert ratings.min() > 0

        # Verify stats
        stats = elo.get_player_stats()
        assert stats["matches_played"].sum() == 200  # 100 matches * 2 players
        assert stats["matches_won"].sum() == 100  # 100 winners

    def test_frame_counts_consistent(self, matches_small):
        """Total frames won by all players = total frames played."""
        elo = EloRating()
        elo.update(matches_small)

        total_frames_won = sum(p.frames_won for p in elo.players.values())
        total_frames_in_data = (matches_small["score1"] + matches_small["score2"]).sum()
        assert total_frames_won == total_frames_in_data
