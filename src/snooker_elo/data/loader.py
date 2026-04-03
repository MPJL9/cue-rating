"""Load and validate match data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _find_default_path() -> Path:
    """Find matches.csv, works both locally and on Render."""
    candidates = [
        Path(__file__).parents[3] / "data" / "raw" / "matches.csv",
        Path.cwd() / "data" / "raw" / "matches.csv",
        Path("/opt/render/project/src/data/raw/matches.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Fall back to first for error message


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    """Load matches.csv and return a validated DataFrame.

    Args:
        path: Path to matches.csv. Defaults to data/raw/matches.csv.

    Returns:
        DataFrame with columns [player1, player2, score1, score2,
        best_of, tournament_id, date, year], sorted by tournament order.
    """
    if path is None:
        path = _find_default_path()

    df = pd.read_csv(
        path,
        dtype={
            "player1": str,
            "player2": str,
            "score1": int,
            "score2": int,
            "best_of": int,
            "tournament_id": str,
            "year": int,
        },
    )

    required = {"player1", "player2", "score1", "score2", "best_of", "tournament_id", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def get_tournament_ids(matches: pd.DataFrame) -> list[str]:
    """Return ordered list of unique tournament IDs (preserving DataFrame order)."""
    return list(matches["tournament_id"].unique())


def split_by_tournament(
    matches: pd.DataFrame, tournament_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split matches into (before tournament, in tournament).

    Returns:
        (matches_before, matches_in_tournament)
    """
    ids = get_tournament_ids(matches)
    idx = ids.index(tournament_id)
    before_ids = set(ids[:idx])

    mask_before = matches["tournament_id"].isin(before_ids)
    mask_in = matches["tournament_id"] == tournament_id

    return matches[mask_before].copy(), matches[mask_in].copy()
