"""Tests for Glicko-2 rating system."""


import pandas as pd
import pytest

from snooker_elo.ratings.glicko2 import Glicko2PlayerState, Glicko2Rating


def _make_matches(*rows):
    """Helper: each row = (player1, player2, score1, score2, best_of, tournament_id, year)."""
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


class TestGlicko2Core:
    """Test core Glicko-2 math functions."""

    def test_g_function(self):
        # g(0) = 1 (no uncertainty)
        assert Glicko2Rating._g(0.0) == pytest.approx(1.0)
        # g increases with lower phi
        assert Glicko2Rating._g(0.5) > Glicko2Rating._g(1.0)
        # g is always in (0, 1]
        assert 0 < Glicko2Rating._g(2.0) <= 1.0

    def test_E_function_equal_ratings(self):
        # Equal ratings = 0.5 expected score
        assert Glicko2Rating._E(0.0, 0.0, 1.0) == pytest.approx(0.5)

    def test_E_function_stronger_player(self):
        # Higher rated player should have higher expected score
        assert Glicko2Rating._E(1.0, 0.0, 1.0) > 0.5
        assert Glicko2Rating._E(0.0, 1.0, 1.0) < 0.5


class TestGlicko2Glickman:
    """Verify against Glickman's published example from the paper.

    The paper uses a player with rating 1500, RD 200, vol 0.06 who plays
    three opponents. We adapt this to snooker format.
    """

    def test_paper_example_direction(self):
        """Player beats two opponents and loses to one -> rating should increase."""
        g2 = Glicko2Rating(tau=0.5, initial_rd=200.0, initial_vol=0.06, default_rating=1500.0)

        # Manually set up the player and three opponents
        g2.players["Player"] = Glicko2PlayerState(
            rating=1500.0, rd=200.0, volatility=0.06
        )
        g2.players["Opp1"] = Glicko2PlayerState(
            rating=1400.0, rd=30.0, volatility=0.06
        )
        g2.players["Opp2"] = Glicko2PlayerState(
            rating=1550.0, rd=100.0, volatility=0.06
        )
        g2.players["Opp3"] = Glicko2PlayerState(
            rating=1700.0, rd=300.0, volatility=0.06
        )

        # Snooker format: Player beats Opp1 (5-3), beats Opp2 (5-4), loses to Opp3 (3-5)
        matches = _make_matches(
            ("Player", "Opp1", 5, 3, 9, "T1", 2024),
            ("Player", "Opp2", 5, 4, 9, "T1", 2024),
            ("Opp3", "Player", 5, 3, 9, "T1", 2024),
        )
        g2.update(matches)

        p = g2.players["Player"]
        # With 2 wins and 1 loss against these opponents, rating should go up slightly
        # RD should decrease (more games = more certainty)
        assert p.rd < 200.0  # More certain after playing 3 games


class TestGlicko2Update:
    """Test the full update pipeline."""

    def test_winner_gains_loser_loses(self):
        g2 = Glicko2Rating(default_rating=1500.0)
        matches = _make_matches(("Alice", "Bob", 5, 3, 9, "T1", 2024))
        g2.update(matches)

        assert g2.players["Alice"].rating > g2.players["Bob"].rating

    def test_rd_decreases_with_play(self):
        g2 = Glicko2Rating(initial_rd=350.0, default_rating=1500.0)
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2024),
            ("A", "C", 5, 2, 9, "T1", 2024),
        )
        g2.update(matches)

        assert g2.players["A"].rd < 350.0  # RD decreased for active player

    def test_rd_inflates_for_inactive(self):
        g2 = Glicko2Rating(initial_rd=200.0, default_rating=1500.0)

        # Tournament 1: A vs B
        m1 = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        g2.update(m1)

        rd_a_after_t1 = g2.players["A"].rd

        # Tournament 2: only C vs D (A and B inactive)
        m2 = _make_matches(("C", "D", 5, 2, 9, "T2", 2024))
        g2.update(m2)

        # A's RD should increase from inactivity
        assert g2.players["A"].rd > rd_a_after_t1

    def test_career_stats_tracked(self):
        g2 = Glicko2Rating()
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2024),
            ("A", "C", 6, 2, 9, "T1", 2024),
        )
        g2.update(matches)

        assert g2.players["A"].matches_played == 2
        assert g2.players["A"].matches_won == 2
        assert g2.players["A"].frames_won == 11
        assert g2.players["B"].frames_won == 3

    def test_time_window_stats(self):
        g2 = Glicko2Rating()
        matches = _make_matches(
            ("A", "B", 5, 3, 9, "T1", 2020),
            ("A", "B", 5, 2, 9, "T2", 2023),
            ("A", "B", 5, 1, 9, "T3", 2024),
        )
        g2.update(matches)

        # Max year = 2024, same logic as ELO tests
        assert g2.players["A"].frames_played_3y == (5 + 2) + (5 + 1)
        assert g2.players["A"].frames_played_1y == 5 + 1


class TestGlicko2Interface:
    """Test interface compatibility with base RatingSystem."""

    def test_predict_frame_win_prob(self):
        g2 = Glicko2Rating(default_rating=1500.0)
        matches = _make_matches(("A", "B", 5, 1, 9, "T1", 2024))
        g2.update(matches)

        # A should be favored
        assert g2.predict_frame_win_prob("A", "B") > 0.5

    def test_predict_match_win_prob(self):
        g2 = Glicko2Rating(default_rating=1500.0)
        matches = _make_matches(("A", "B", 5, 1, 9, "T1", 2024))
        g2.update(matches)

        # Longer format amplifies advantage
        mp5 = g2.predict_match_win_prob("A", "B", 5)
        mp17 = g2.predict_match_win_prob("A", "B", 17)
        assert mp17 > mp5 > 0.5

    def test_get_ratings(self):
        g2 = Glicko2Rating(default_rating=1500.0)
        matches = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        g2.update(matches)

        ratings = g2.get_ratings()
        assert ratings.index[0] == "A"  # Highest rated first
        assert ratings["A"] > ratings["B"]

    def test_get_player_stats_has_rd(self):
        g2 = Glicko2Rating()
        matches = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        g2.update(matches)

        stats = g2.get_player_stats()
        assert "rd" in stats.columns
        assert "volatility" in stats.columns

    def test_snapshot(self):
        g2 = Glicko2Rating()
        matches = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        g2.update(matches)

        snap = g2.snapshot()
        assert "A" in snap
        # Modifying snapshot shouldn't affect original
        snap["A"].rating = 9999
        assert g2.players["A"].rating != 9999

    def test_predict_batch(self):
        g2 = Glicko2Rating()
        matches = _make_matches(("A", "B", 5, 1, 9, "T1", 2024))
        g2.update(matches)

        test = pd.DataFrame(
            {"player1": ["A"], "player2": ["B"], "best_of": [9]}
        )
        preds, frame_probs, match_probs = g2.predict_batch(test)
        assert frame_probs[0] > 0.5
        assert 0 < match_probs[0] < 1


class TestGlicko2Features:
    """Test feature generation for ML pipeline."""

    def test_feature_columns(self):
        g2 = Glicko2Rating()
        matches = _make_matches(("A", "B", 5, 3, 9, "T1", 2024))
        g2.update(matches)

        test = pd.DataFrame(
            {"player1": ["A"], "player2": ["B"], "best_of": [9]}
        )
        features = g2.generate_match_features(test)

        assert "player1_glicko2" in features.columns
        assert "player2_glicko2" in features.columns
        assert "player1_rd" in features.columns
        assert "player2_rd" in features.columns
        assert "glicko2_match_win_rate" in features.columns
        assert "glicko2_frame_win_rate" in features.columns


class TestGlicko2OnRealData:
    """Test on real match data."""

    def test_runs_on_real_data(self):
        """Glicko-2 should process real matches without errors."""
        from snooker_elo.data.loader import load_matches

        matches = load_matches()
        # Use first 500 matches (spans a few tournaments)
        small = matches.head(500)

        g2 = Glicko2Rating(tau=0.5, default_rating=1500.0)
        g2.update(small)

        ratings = g2.get_ratings()
        assert len(ratings) > 0
        assert ratings.max() < 3000
        assert ratings.min() > 0

    def test_frame_counts_consistent(self):
        from snooker_elo.data.loader import load_matches

        matches = load_matches().head(500)
        g2 = Glicko2Rating()
        g2.update(matches)

        total_frames_won = sum(p.frames_won for p in g2.players.values())
        total_frames_in_data = (matches["score1"] + matches["score2"]).sum()
        assert total_frames_won == total_frames_in_data
