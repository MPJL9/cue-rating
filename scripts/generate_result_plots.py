"""Generate result plots for the README and writeup.

Creates:
1. three_system_comparison.png — ELO vs Glicko-2 vs Bayesian on elite-vs-elite
2. rankings_benchmark.png — Our model vs World Rankings
3. bayesian_skill_intervals.png — Top 20 players with credible intervals
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Dark theme matching the web app
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "#0a0a0a",
    "axes.facecolor": "#111113",
    "axes.edgecolor": "#333",
    "grid.color": "#222",
    "grid.alpha": 0.5,
})

GREEN = "#22c55e"
BLUE = "#3b82f6"
PURPLE = "#a855f7"
GRAY = "#71717a"
ORANGE = "#f97316"
YELLOW = "#eab308"

OUT = "data/processed"


# ─────────────────────────────────────────────────────────────────────
# 1. Three-system comparison (ELO vs Glicko-2 vs Bayesian)
# ─────────────────────────────────────────────────────────────────────
def plot_three_systems():
    # Hardcoded from benchmark_three_systems.py output (983 elite-vs-elite matches)
    systems = ["ELO\n(full history)", "Glicko-2\n(full history)", "Bayesian BT\n(recent only)"]
    accuracy = [0.5554, 0.5788, 0.6297]
    log_loss = [0.7123, 0.6979, 0.6547]
    brier = [0.2559, 0.2483, 0.2314]
    colors = [BLUE, PURPLE, GREEN]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Accuracy (higher is better)
    bars = axes[0].bar(systems, accuracy, color=colors, edgecolor="#333")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy ↑", color="white")
    axes[0].set_ylim(0.5, 0.7)
    axes[0].axhline(0.5, color="#666", linestyle="--", linewidth=0.8, label="Coin flip")
    for bar, val in zip(bars, accuracy):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                     f"{val*100:.1f}%", ha="center", fontweight="bold")

    # Log loss (lower is better)
    bars = axes[1].bar(systems, log_loss, color=colors, edgecolor="#333")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title("Log Loss ↓", color="white")
    axes[1].set_ylim(0.5, 0.75)
    for bar, val in zip(bars, log_loss):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                     f"{val:.3f}", ha="center", fontweight="bold")

    # Brier (lower is better)
    bars = axes[2].bar(systems, brier, color=colors, edgecolor="#333")
    axes[2].set_ylabel("Brier Score")
    axes[2].set_title("Brier Score ↓", color="white")
    axes[2].set_ylim(0.18, 0.27)
    for bar, val in zip(bars, brier):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                     f"{val:.3f}", ha="center", fontweight="bold")

    fig.suptitle(
        "Three Rating Systems on 983 Elite-vs-Elite Matches",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/three_system_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.close()
    print("Saved three_system_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# 2. Rankings benchmark
# ─────────────────────────────────────────────────────────────────────
def plot_rankings_benchmark():
    with open(f"{OUT}/benchmark_results.json") as f:
        data = json.load(f)

    methods_order = ["coin_flip", "world_rankings", "pure_elo", "our_model"]
    labels = ["Coin Flip\n(baseline)", "World Rankings\n(public)", "Pure ELO\n(ours)", "ML Model\n(ours)"]
    colors = [GRAY, ORANGE, BLUE, GREEN]

    accuracy = [data["methods"][m]["accuracy"] for m in methods_order]
    log_loss = [data["methods"][m]["log_loss"] for m in methods_order]
    brier = [data["methods"][m]["brier"] for m in methods_order]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    bars = axes[0].bar(labels, accuracy, color=colors, edgecolor="#333")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy ↑", color="white")
    axes[0].set_ylim(0.45, 0.75)
    axes[0].axhline(0.5, color="#666", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, accuracy):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.008,
                     f"{val*100:.1f}%", ha="center", fontweight="bold", fontsize=10)

    bars = axes[1].bar(labels, log_loss, color=colors, edgecolor="#333")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title("Log Loss ↓", color="white")
    for bar, val in zip(bars, log_loss):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                     f"{val:.3f}", ha="center", fontweight="bold", fontsize=10)

    bars = axes[2].bar(labels, brier, color=colors, edgecolor="#333")
    axes[2].set_ylabel("Brier Score")
    axes[2].set_title("Brier Score ↓", color="white")
    for bar, val in zip(bars, brier):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                     f"{val:.3f}", ha="center", fontweight="bold", fontsize=10)

    fig.suptitle(
        f"Our Model vs Public Baselines on {data['test_size']} Held-Out Matches (2015-2019)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f"{OUT}/rankings_benchmark.png", dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.close()
    print("Saved rankings_benchmark.png")


# ─────────────────────────────────────────────────────────────────────
# 3. Bayesian skill intervals
# ─────────────────────────────────────────────────────────────────────
def plot_bayesian_intervals():
    with open("data/raw/bayesian_bt_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    df = cache["summary_df"].head(20).iloc[::-1]  # Reverse for top-down display

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(df))

    # Credible intervals as horizontal lines
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot(
            [row["skill_q025"], row["skill_q975"]], [i, i],
            color=BLUE, linewidth=2, alpha=0.7,
        )
        ax.scatter(
            [row["skill_mean"]], [i],
            color=GREEN, s=80, zorder=5, edgecolor="white", linewidth=1,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["player"])
    ax.set_xlabel("Bayesian Skill (posterior)")
    ax.set_title(
        "Top 20 Players by Bayesian Skill — 95% Credible Intervals",
        fontsize=13, fontweight="bold",
    )
    ax.axvline(0, color="#444", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN,
               markersize=10, label="Posterior mean", markeredgecolor="white"),
        Line2D([0], [0], color=BLUE, linewidth=2, label="95% credible interval"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{OUT}/bayesian_skill_intervals.png", dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.close()
    print("Saved bayesian_skill_intervals.png")


# ─────────────────────────────────────────────────────────────────────
# 4. Top players ELO + Glicko-2 chart
# ─────────────────────────────────────────────────────────────────────
def plot_top_players():
    from snooker_elo.data.loader import load_matches
    from snooker_elo.ratings.elo import EloRating
    from snooker_elo.ratings.glicko2 import Glicko2Rating

    matches = load_matches()
    elo = EloRating(k_factor=9.77, divisor=327.15)
    elo.update(matches)
    g2 = Glicko2Rating(tau=1.488, default_rating=1500)
    g2.update(matches)

    # Top 15 by ELO
    elo_top = elo.get_ratings().head(15)
    names = list(elo_top.index)
    elo_vals = list(elo_top.values)
    g2_vals = [g2.players[n].rating if n in g2.players else 1500 for n in names]

    # Normalize Glicko-2 to roughly the same scale for visual comparison
    # Glicko-2 is centered at 1500, ELO at 1000 — shift to align top player
    g2_shift = elo_vals[0] - g2_vals[0]
    g2_normalized = [v + g2_shift for v in g2_vals]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(names))[::-1]
    width = 0.4

    ax.barh(y + width/2, elo_vals, width, color=GREEN, label="ELO", edgecolor="#333")
    ax.barh(y - width/2, g2_normalized, width, color=BLUE,
            label="Glicko-2 (shifted to align top)", edgecolor="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Rating")
    ax.set_title(
        "Top 15 Players: ELO vs Glicko-2 (April 2026)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(min(min(elo_vals), min(g2_normalized)) - 50, max(elo_vals) + 50)

    plt.tight_layout()
    plt.savefig(f"{OUT}/top_players.png", dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.close()
    print("Saved top_players.png")


if __name__ == "__main__":
    plot_three_systems()
    plot_rankings_benchmark()
    plot_bayesian_intervals()
    plot_top_players()
    print("\nAll plots saved to data/processed/")
