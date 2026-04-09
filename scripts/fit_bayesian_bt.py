"""Fit a Bayesian Bradley-Terry model for snooker player skills.

Model:
    skill_i  ~ Normal(0, 1)         # Standard normal prior
    p_ij     = sigmoid(s_i - s_j)   # P(player i wins a frame vs j)
    f_ij     ~ Binomial(n_ij, p_ij) # Number of frames i wins out of n_ij total

Restricted to active players (>=100 matches) using last 2 years of data.

Why this model:
- Fully Bayesian — full posterior over each player's skill, not just point estimate
- Hierarchical structure (could be extended to per-tournament random effects)
- Incorporates uncertainty: a player with 5 matches has a wider posterior than 500 matches
"""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import pymc as pm

from snooker_elo.data.loader import load_matches


def prepare_data(matches: pd.DataFrame, min_matches: int = 100, recent_years: int = 2):
    """Aggregate matches into frame counts per player pair.

    Returns: (player_to_idx, pair_data)
    pair_data is a list of dicts: [{i, j, wins_i, wins_j}, ...]
    """
    max_year = int(matches["year"].max())
    cutoff = max_year - recent_years
    recent = matches[matches["year"] >= cutoff].copy()

    # Filter to players with enough recent matches
    p_count = pd.concat([recent["player1"], recent["player2"]]).value_counts()
    eligible = set(p_count[p_count >= min_matches].index)
    recent = recent[
        recent["player1"].isin(eligible) & recent["player2"].isin(eligible)
    ].copy()

    # Drop draws
    recent = recent[recent["score1"] != recent["score2"]].copy()

    print(f"  Filtered to {len(recent)} matches between {len(eligible)} active players")

    # Build player index
    players = sorted(eligible)
    p2i = {p: i for i, p in enumerate(players)}

    # Aggregate frame outcomes per ordered pair (i < j)
    # This is more efficient than treating each match as a separate observation
    pair_wins = {}  # (i, j) -> [wins_i, wins_j], i < j
    for _, row in recent.iterrows():
        p1, p2 = row["player1"], row["player2"]
        s1, s2 = int(row["score1"]), int(row["score2"])
        i, j = p2i[p1], p2i[p2]
        if i > j:
            i, j = j, i
            s1, s2 = s2, s1
        key = (i, j)
        if key not in pair_wins:
            pair_wins[key] = [0, 0]
        pair_wins[key][0] += s1
        pair_wins[key][1] += s2

    # Convert to arrays for PyMC
    pairs = list(pair_wins.keys())
    i_arr = np.array([p[0] for p in pairs], dtype=np.int32)
    j_arr = np.array([p[1] for p in pairs], dtype=np.int32)
    wins_i = np.array([pair_wins[p][0] for p in pairs], dtype=np.int32)
    wins_j = np.array([pair_wins[p][1] for p in pairs], dtype=np.int32)
    n_total = wins_i + wins_j

    print(f"  {len(pairs)} unique player pairs, {n_total.sum()} total frames")
    return players, p2i, i_arr, j_arr, wins_i, n_total


def fit_model(players, i_arr, j_arr, wins_i, n_total, draws=1000, tune=500):
    """Fit Bayesian Bradley-Terry model with NUTS."""
    n_players = len(players)
    print(f"\nBuilding model for {n_players} players...")

    with pm.Model() as model:
        # Wider prior: Normal(0, 2) — less informative, lets data dominate
        skill_raw = pm.Normal("skill_raw", mu=0.0, sigma=2.0, shape=n_players)
        # Center skills (sum to zero) — breaks rotational invariance
        skill = pm.Deterministic("skill", skill_raw - pm.math.mean(skill_raw))
        diff = skill[i_arr] - skill[j_arr]
        p = pm.math.sigmoid(diff)
        pm.Binomial("obs", n=n_total, p=p, observed=wins_i)

        print(f"Sampling: {draws} draws, {tune} tune, 4 chains...")
        start = time.time()
        trace = pm.sample(
            draws=draws, tune=tune, chains=4, cores=4,
            target_accept=0.9, progressbar=True, random_seed=42,
        )
        print(f"Sampling done in {time.time()-start:.0f}s")

    return trace


def main():
    print("Loading data...")
    matches = load_matches()
    print(f"  {len(matches)} total matches")

    players, p2i, i_arr, j_arr, wins_i, n_total = prepare_data(
        matches, min_matches=100, recent_years=2
    )

    trace = fit_model(players, i_arr, j_arr, wins_i, n_total, draws=2000, tune=1500)

    # Extract posterior samples
    posterior = trace.posterior["skill"].values  # (chains, draws, players)
    posterior = posterior.reshape(-1, len(players))  # (samples, players)

    # Posterior summary
    means = posterior.mean(axis=0)
    stds = posterior.std(axis=0)
    q025 = np.quantile(posterior, 0.025, axis=0)
    q975 = np.quantile(posterior, 0.975, axis=0)

    df = pd.DataFrame({
        "player": players,
        "skill_mean": means,
        "skill_std": stds,
        "skill_q025": q025,
        "skill_q975": q975,
    }).sort_values("skill_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    print("\nTop 15 by Bayesian skill (mean):")
    print(df.head(15).to_string(index=False))

    # Save
    out = {
        "players": players,
        "posterior_samples": posterior.astype(np.float32),  # (n_samples, n_players)
        "summary_df": df,
    }
    with open("data/raw/bayesian_bt_cache.pkl", "wb") as f:
        pickle.dump(out, f)

    import os
    print(f"\nSaved to data/raw/bayesian_bt_cache.pkl ({os.path.getsize('data/raw/bayesian_bt_cache.pkl')/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
