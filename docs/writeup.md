# Rating Systems and Match Prediction for Professional Snooker: A Comparative Study

**Author:** Tianxiang Liu
**Project repository:** https://github.com/MPJL9/cue-rating
**Live demo:** https://cue-rating.onrender.com
**Date:** April 2026

---

## Abstract

We adapt two probabilistic rating systems — ELO (Elo, 1978) and Glicko-2 (Glickman, 2001) — to professional snooker, and evaluate them on 117,530 matches spanning 1982 to 2026. Unlike standard sports applications which model match outcomes directly, our system models *frame win probability* and derives match outcomes via the binomial distribution, exploiting snooker's structure where matches consist of multiple independent frames. We optimize all parameters via Maximum Likelihood Estimation rather than grid search. Both rating systems alone achieve ~68% match prediction accuracy on a held-out test set of 6,905 matches. Combining them with light gradient boosting on six features (rating predictions plus uncertainty estimates) yields **70.2% accuracy**, well-calibrated probabilistic predictions (Expected Calibration Error 0.011), and a Brier score of 0.192. We find that the two rating systems' match win probabilities account for 74.5% of the predictive power; player statistics, head-to-head records, and rating momentum contribute marginally. We discuss the implications of this finding for the practical ceiling of pre-match snooker prediction.

---

## 1. Introduction

Rating systems convert pairwise competition outcomes into a single number per competitor, enabling ranking and prediction. The Elo system, originally developed for chess (Elo, 1978), assumes a competitor's "true skill" follows a normal (or logistic) distribution and updates ratings based on the difference between observed and expected outcomes. Glicko (Glickman, 1995) and its successor Glicko-2 (Glickman, 2001) extend Elo with explicit uncertainty estimation and volatility tracking.

Snooker presents an interesting application: matches are decided by frames (individual games), and the number of frames varies dramatically across competitions — from Best-of-5 in early-round qualifiers to Best-of-35 in the World Championship final. A skilled player who wins 56% of frames has a vastly different chance of winning a Best-of-5 match (≈58%) versus a Best-of-35 final (≈75%). A rating system that models *match outcomes* directly cannot transfer information across formats; one that models *frame outcomes* can.

We make the following contributions:

1. We adapt ELO and Glicko-2 to model frame win probability with frame-weighted updates, enabling format-agnostic match prediction via the binomial distribution.
2. We optimize all rating system parameters by maximizing the log-likelihood of observed frame-level outcomes — a principled alternative to manual tuning.
3. We compare both systems on 117,530 professional snooker matches across multiple metrics: accuracy, log loss, Brier score, and expected calibration error.
4. We train classification models combining the two rating systems and additional features, and analyze the marginal value of each feature category.
5. We deploy the system as an interactive web application supporting live rankings, head-to-head prediction, Monte Carlo bracket simulation, and historical analysis.

---

## 2. Data

### 2.1 Source

Match data was obtained from two sources:
- **1982–2020:** A Kaggle dataset by user `rusiano` derived from cuetracker.net.
- **2020–2026:** Direct scraping of cuetracker.net using a polite crawler (1.5 second delay between requests, retry with exponential backoff).

After cleaning — removing walkover matches, amateur and pro-am events, and matches with malformed scores — the combined dataset contains:

| Statistic | Value |
|-----------|-------|
| Total matches | 117,530 |
| Total players | 3,849 |
| Total tournaments | 1,108 |
| Date range | 1982–2026 |

Each match record contains: `player1`, `player2`, `score1`, `score2`, `best_of`, `tournament_id`, `date`, `year`. Player1 is conventionally the winner; we randomize this during ML training to remove positional bias.

### 2.2 Held-Out Test Set

For model evaluation we use the most recent 300 tournaments (34,521 matches) and apply a temporal 80/20 split: the first 27,616 matches train the models, the last 6,905 form the test set. The split is strictly chronological — no future information leaks into training.

---

## 3. Rating Systems

### 3.1 Frame-Based ELO

We model the probability that player 1 wins a single frame against player 2 with the standard Elo logistic function:

```
E_1 = 1 / (1 + exp((R_2 - R_1) / d))
```

where `d` is a scaling parameter (the conventional value 400 is replaced with our MLE-optimized value below). After observing a match with frame scores `s_1` and `s_2`, we update both ratings:

```
R_new = R + K · (s_1 + s_2) · (S - E)
```

where `K` is the adjustment rate, `S = s_1 / (s_1 + s_2)` is the actual frame win rate, and `E` is the expected frame win rate. The factor `(s_1 + s_2)` weights the update by total frames played: longer matches provide more information and produce larger rating swings.

The match win probability for a Best-of-`N` format is computed by summing binomial probabilities over winning scorelines:

```
P(match win) = Σ_{k=w}^{N} C(k-1, w-1) · p^(w-1) · (1-p)^(k-w) · p
```

where `w = ⌈N/2⌉` is the number of frames needed to win.

### 3.2 Glicko-2

Glicko-2 maintains three quantities per player: rating (μ), rating deviation (φ, abbreviated RD), and volatility (σ). The update procedure (Glickman, 2001) consists of converting to Glicko-2 scale, computing the variance and performance delta from observed outcomes, iteratively solving for the new volatility via the Illinois algorithm, and updating φ and μ.

We adapt Glicko-2 to snooker as follows:
- **Rating period = tournament**: All matches in a tournament are processed as a single rating period. Between tournaments, RDs of inactive players inflate according to the standard formula.
- **Frame-weighted continuous outcomes**: Instead of binary win/loss, we use the frame win proportion `s = s_1 / (s_1 + s_2)` as the outcome, weighted by total frames played. This parallels the frame-based ELO update.

### 3.3 Bayesian Bradley-Terry

ELO and Glicko-2 are both implicitly fitting variants of the Bradley-Terry model (Bradley & Terry, 1952), which assigns each player a latent skill `s` and models the probability that player *i* beats player *j* as `sigmoid(s_i - s_j)`. ELO is essentially online stochastic gradient descent on the Bradley-Terry log-likelihood; Glicko-2 is an approximate Bayesian extension that maintains a Gaussian belief over each player's skill.

We implement a *fully* Bayesian version using PyMC and MCMC inference. The model is:

```
s_i ~ Normal(0, 2)        for each player i
p_ij = sigmoid(s_i - s_j)  predicted frame win prob
f_ij ~ Binomial(n_ij, p_ij)  observed frames i won out of n_ij vs j
```

We aggregate frame counts per ordered player pair and fit the joint posterior over all skills using the No-U-Turn Sampler (Hoffman & Gelman, 2014) — 4 chains × 2000 draws after 1500 tuning steps. To break the model's rotational invariance (the likelihood is unchanged under a uniform shift of all skills), we center the skills to sum to zero via a deterministic transformation.

We restrict the Bayesian fit to **active players** (≥100 matches in the past 2 years), yielding 29 elite professionals. This restriction is necessary for two reasons: MCMC over all 3,849 players would be computationally prohibitive, and players with few recent matches would have posteriors essentially identical to the prior — providing no useful information.

Sampling completes in ~5 seconds. The output is a posterior with 8000 samples per player; predictions for new matches use **posterior predictive averaging**:

```
P(i beats j in a frame) ≈ (1/N) · Σ_k sigmoid(s_i^(k) - s_j^(k))
```

This properly propagates skill uncertainty through to prediction uncertainty — a feature ELO and Glicko-2 cannot offer.

A standalone document at [`docs/bayesian_bt_explained.md`](bayesian_bt_explained.md) walks through the derivation in detail.

### 3.4 Parameter Optimization via MLE

Treating each frame as a Bernoulli trial with probability `p_i` predicted by the rating system, the log-likelihood of observed frame scores across all matches is:

```
L = Σ_i [ s_{1,i} · log(p_i) + s_{2,i} · log(1 - p_i) ]
```

We maximize `L` over the parameter space using the Nelder-Mead simplex algorithm (`scipy.optimize.minimize`). The full match history is replayed under each candidate parameter set, which is feasible because our optimized rating engine processes 117,530 matches in approximately 0.3 seconds (ELO) and 2.1 seconds (Glicko-2).

#### Optimized parameters

| System | Parameter | Conventional | MLE-Optimized |
|--------|-----------|--------------|---------------|
| ELO | K-factor | 8.0 (Erdos baseline) | **9.77** |
| ELO | Divisor | 400 | **327.15** |
| Glicko-2 | τ (volatility constraint) | 0.5 | **1.488** |

The optimized divisor of 327 versus the chess convention of 400 reflects snooker's lower per-frame skill differential — small rating gaps translate to smaller frame win probabilities than in chess.

---

## 4. Match Prediction Models

We train three classifiers on five feature sets:

| Feature set | Features | Description |
|-------------|----------|-------------|
| ELO only | 2 | `elo_frame_win_rate`, `elo_match_win_rate` |
| Glicko-2 only | 4 | Glicko-2 frame/match probabilities + both players' RD |
| Ratings combined | 6 | All ELO and Glicko-2 outputs |
| Clean (15) | 15 | Above + win rate differences, form, h2h, momentum |
| All raw (35) | 35 | Above + raw counts (matches played, frames won, etc.) |

Models: Logistic Regression (with feature scaling), Random Forest (n=200, max_depth=6), and Gradient Boosting (n=100, max_depth=5, learning rate 0.05).

### 4.1 Results

| Model | Features | Accuracy | Log Loss | Brier | ECE |
|-------|----------|----------|----------|-------|-----|
| **Gradient Boosting** | **Ratings combined (6)** | **0.7024** | **0.5623** | **0.1918** | **0.0111** |
| Logistic Regression | All features (35) | 0.7005 | 0.5650 | 0.1921 | 0.0201 |
| Logistic Regression | Ratings combined (6) | 0.6993 | 0.5678 | 0.1931 | 0.0210 |
| Gradient Boosting | All features (35) | 0.6989 | 0.5628 | 0.1917 | 0.0143 |
| Gradient Boosting | Clean features (15) | 0.6983 | 0.5639 | 0.1921 | 0.0158 |
| Gradient Boosting | Glicko-2 only (4) | 0.6965 | 0.5624 | 0.1918 | 0.0139 |
| Random Forest | Glicko-2 only (4) | 0.6954 | 0.5657 | 0.1930 | 0.0138 |
| Logistic Regression | ELO only (2) | 0.6873 | 0.5845 | 0.2003 | 0.0169 |
| Pure ELO (no ML) | — | 0.6839 | 0.5863 | 0.2013 | 0.0112 |
| Pure Glicko-2 (no ML) | — | 0.6798 | 0.5918 | 0.2024 | 0.0494 |
| Coin flip baseline | — | 0.5000 | 0.6931 | 0.2500 | 0.0000 |

The best model is **Gradient Boosting on six rating features**, achieving 70.24% accuracy. Notably, this uses only the predictions of ELO and Glicko-2 plus the two players' rating deviations — adding 29 more features (raw stats, head-to-head, momentum, inactivity) yields no improvement and sometimes hurts performance.

### 4.2 Feature Importance

Feature importances from the Gradient Boosting model on the 35-feature set:

| Feature | Importance |
|---------|-----------|
| `elo_match_win_rate` | 0.457 |
| `glicko2_match_win_rate` | 0.289 |
| `player1_rd` | 0.035 |
| `player2_rd` | 0.028 |
| `elo_frame_win_rate` | 0.027 |
| `glicko2_frame_win_rate` | 0.020 |
| `match_wr_diff` | 0.012 |
| `form_1y_diff` | 0.011 |
| `momentum_diff` | 0.010 |
| All other 26 features (sum) | 0.111 |

The two rating-system match win probabilities account for **74.5% of the model's total feature importance**. Glicko-2's RD parameter — uniquely available among rating-system outputs — contributes another 6.3%, suggesting that uncertainty quantification adds value beyond point estimates of skill.

### 4.3 Comparison Against the Official World Rankings

The most natural baseline for any sports prediction model is the official ranking system that the sport's governing body publishes. World Snooker Tour rankings are based on prize money earned over the prior two seasons and are heavily referenced by bookmakers when setting odds. We compare four approaches on a held-out test set of 4,777 matches from the 2015–2019 period (where both modern data and ranking information are available):

| Method | Accuracy | Log Loss | Brier Score |
|--------|----------|----------|-------------|
| Coin flip | 0.5000 | 0.6931 | 0.2500 |
| World Rankings (rank-difference) | 0.6351 | 0.6580 | 0.2285 |
| Pure ELO (threshold) | 0.6831 | 0.5882 | 0.2017 |
| **Our model (Gradient Boosting + 6 features)** | **0.6875** | **0.5788** | **0.1981** |

The world ranking baseline uses the prior season's ranking and predicts via a soft sigmoid on the log rank ratio. Coverage is 41% of test matches (rankings often miss recent professionals or qualifiers); for matches without ranking data, we fall back to a 0.5 prior.

**Our model beats the official rankings by 5.2 percentage points in accuracy and reduces log loss by 12%**. Pure ELO alone beats rankings by 4.8 points, suggesting that even a simple frame-weighted Elo update extracts more signal than the ranking points formula. The marginal gain from our full ML pipeline over pure ELO is small but consistent across all three metrics, primarily attributable to the Glicko-2 RD feature.

These results are conservative for two reasons:
1. We use only the prior season's ranking, while bookmakers update odds with current information.
2. The 2015–2019 window is relatively easier to predict than post-2019 (more established players, fewer qualifiers).

Bookmakers typically reach 67–70% accuracy on individual sports (Kovalchik, 2016 — tennis), so our 68.8% is competitive with what professional odds-makers achieve, though a direct head-to-head comparison would require a paid odds dataset.

### 4.4 Three-System Comparison on Elite Matches

We evaluate ELO, Glicko-2, and Bayesian Bradley-Terry on the same held-out test set restricted to matches between 29 elite players (≥100 matches in the past 2 years), giving 983 matches:

| System | Accuracy | Log Loss | Brier |
|--------|----------|----------|-------|
| ELO (full history) | 0.5554 | 0.7123 | 0.2559 |
| Glicko-2 (full history) | 0.5788 | 0.6979 | 0.2483 |
| **Bayesian BT (recent only)** | **0.6297** | **0.6547** | **0.2314** |

The Bayesian model wins on all three metrics. Two factors contribute:

1. **Recent data only.** The Bayesian fit uses the past 2 years, while ELO and Glicko-2 carry information from all of 1982-onward. For elite players with shifting form, recent data is more relevant.
2. **Posterior predictive averaging.** Even when point estimates agree, averaging predictions across 8000 posterior samples produces better-calibrated probabilities than single-point predictions.

Note that on the broader test set (Section 4.1, including matches with one or both players outside the elite group), the Bayesian model cannot make predictions and so its accuracy advantage doesn't extend automatically. The classical 70.2% from gradient boosting on rating-system features remains the best general-purpose model.

The Bayesian extension is most valuable as a *probabilistic modeling exercise* — it demonstrates principled uncertainty quantification, which is the foundation for any Bayesian decision-making in trading, recommendation, or active learning.

### 4.5 Why the Elite-vs-Elite Accuracy Looks Lower

A natural question: if the broad-test-set best model achieves 70.2% accuracy (Section 4.1), why does the elite-vs-elite comparison only reach 63.0%? The answer is that **these are not the same prediction problem**, and the elite restriction makes the task strictly harder.

**The broad test set is dominated by easy matches.** Of the 6,905 matches in Section 4.1, the vast majority involve at least one player whose rating is far from the other's. Consider three representative match types from the dataset:

| Match type | Skill gap | Frame win prob | Match win prob (BO9) |
|------------|-----------|----------------|----------------------|
| Top pro vs Q School qualifier | ~600 ELO | ~95% | ~99.9% |
| Top pro vs mid-tier pro | ~150 ELO | ~63% | ~74% |
| Top 5 vs Top 5 | ~5 ELO | ~50.4% | ~50.6% |

The first type is trivially predictable — predict the higher-rated player and you are right almost every time. These easy matches account for a large fraction of the broad test set and inflate aggregate accuracy. The third type is the hardest: when two equally-rated players meet, the outcome is essentially a coin flip and the theoretical ceiling is much lower than 70%.

**The elite restriction throws away the easy matches.** Of 6,134 recent matches, only 983 (16%) have both players in the top 29 active professionals. The remaining 5,151 matches — most of which involve a clear favorite — are excluded. What's left is the hardest possible subset: late-round encounters between players whose skill differences are within the noise floor of any rating system.

**Theoretical ceiling for elite-vs-elite is ~65%.** A simple back-of-envelope: if the average elite-vs-elite frame edge is 53% (Trump 1551 vs Selby 1555 → ~50.5%, Trump vs Higgins → ~54%), then in a Best-of-9 match the favorite wins with probability ~58%. A perfectly calibrated model that always picks the favorite would get ~58% accuracy. The 63% achieved by Bayesian BT exceeds this naive ceiling because it's not just picking the favorite — it's incorporating recent form via the 2-year fitting window.

**Implication for ELO and Glicko-2.** Both fall to 55-58% on this slice for two reasons. First, their cumulative-history ratings drift slowly: Trump's ELO reflects 20 years of average performance, not his current form, so for elite players whose form changes year-to-year, the rating lags. Second, in absolute ELO units, top players cluster within ~80 points of each other, meaning the predicted frame win probabilities are all clustered around 0.5 — essentially noise. The Bayesian fit on recent data only is a fairer reflection of *current* skill differentials.

**The practical takeaway**: a model that scores 63% on elite-vs-elite is closer to the ceiling for that task than a model that scores 70% on the broad task is to its own ceiling. For high-stakes prediction (the matches you actually care about — finals, semis, tournament-deciding fixtures), the Bayesian model is more useful even though its headline accuracy looks lower. This is also why log loss and Brier score are reported alongside accuracy: they degrade gracefully on hard slices and reveal whether the model's *probability estimates* are honest, which matters more than whether it picked the right side of a coin flip.

### 4.6 Calibration

A model is well-calibrated if predicted probabilities correspond to actual frequencies: among matches where the model predicts a 70% win probability, the favored player should win approximately 70% of the time. We measure calibration via Expected Calibration Error (ECE) computed over 10 equal-width probability bins.

The Gradient Boosting model on the 6-feature set achieves ECE = 0.011, meaning predicted probabilities deviate from actual frequencies by only ~1 percentage point on average. Pure Glicko-2 has the worst calibration (ECE = 0.049), indicating overconfident predictions; pure ELO is well-calibrated despite a slightly lower accuracy.

---

## 5. Discussion

### 5.1 The 70% Ceiling

Our best model achieves 70.2% accuracy, and adding more features beyond the two rating-system predictions yields negligible improvement. We interpret this as evidence that **the predictable component of snooker match outcomes is largely captured by long-run player skill differentials**, which both rating systems estimate well. The remaining ~30% error is dominated by irreducible aleatoric variance: short-term form, mental state, fatigue from tournament scheduling, and the inherent stochasticity of cue sport — a missed black ball or a tactical error can swing a frame regardless of skill.

This interpretation is consistent with reported results in similar individual sports. Tennis match prediction with rating systems alone reaches ~67-70%, comparable to bookmaker accuracy (Kovalchik, 2016). Chess match prediction, when restricted to decisive games, plateaus in the 65-69% range. The convergence across sports suggests a common ceiling imposed by day-to-day performance variance rather than model limitations.

### 5.2 Why Glicko-2's RD Matters

Glicko-2's rating deviation captures uncertainty arising from inactivity. A player returning from a long break (e.g., Zhao Xintong's 2023-2024 ban) has high RD even after a few games, reflecting the system's lack of confidence in their current form. ELO has no equivalent — a 1600-rated player who played yesterday is treated identically to one who hasn't played in two years. Including RD as an ML feature gives a 6.3% importance share, second only to the two match win probabilities themselves.

### 5.3 Format Sensitivity

A frame win edge of 56% translates to:
- 58% match win probability in Best-of-5
- 68% in Best-of-17
- 75% in Best-of-35

This is the practical justification for snooker's longer World Championship format: longer matches reduce variance and reward sustained skill over single brilliant performances. Our binomial-derived match win probabilities make this explicit, and the tournament simulator on the live demo uses round-specific best-of values to model real tournament structures (e.g., World Championship: BO19 → BO25 → BO33 → BO35).

### 5.4 Limitations

1. **No direct bookmaker odds comparison**: We benchmark against the official World Rankings as a free public proxy, but a direct comparison against historical bookmaker odds (which incorporate same-day form and injury news) would be the gold standard. Reliable historical snooker odds are gated behind paid APIs.
2. **Pre-match prediction only**: We do not model in-match dynamics (frame-by-frame momentum, comeback probability after going behind).
3. **No surface or venue effects**: Some players historically perform better at specific venues (e.g., the Crucible) or against specific opponents — the model treats all match contexts as equivalent.
4. **Match context features ignored**: Round of the tournament, prize money, ranking points at stake, and player travel/rest are not used.
5. **Symmetric error treatment**: Log-loss treats all matches equally, but predicting an upset correctly is arguably more valuable than confirming a 95% favorite.

---

## 6. Engineering Notes

The system is implemented in Python with the following components:

| Module | Purpose | Key Optimization |
|--------|---------|------------------|
| `ratings/elo.py` | ELO update loop | dict-based player state, ~50–100x faster than DataFrame `.loc[]` |
| `ratings/glicko2.py` | Glicko-2 with Illinois algorithm | tournament-grouped rating periods |
| `ratings/optimization.py` | MLE parameter search | Nelder-Mead via scipy |
| `features/generator.py` | Feature pipeline | O(T·M) incremental, vs O(T²·M) baseline |
| `evaluation/metrics.py` | Metrics + calibration | log-loss, Brier, ECE |
| `web/app.py` | FastAPI backend | precomputed cache loaded in 0.12s |
| `frontend/` | React + Recharts UI | served as static files from FastAPI |

Performance summary (all timings on a 2021 MacBook Pro):

| Operation | Time |
|-----------|------|
| Process 117,530 matches with ELO | 0.27s |
| Process 117,530 matches with Glicko-2 | 2.21s |
| Generate 34,521 features for last 300 tournaments | 5.7s |
| MLE-optimize ELO parameters | ~135s |
| Web app cold start (with cache) | 0.12s |

The deployed application caches all rating computations to a 1.2 MB pickle file checked into the repository. On Render's free tier, the server starts in under 1 second and serves the React frontend from the same FastAPI process — no separate frontend deployment required.

---

## 7. Conclusion

We presented a comparative study of two rating systems applied to professional snooker, demonstrating that frame-level modeling combined with binomial-derived match probabilities is well-suited to the sport's variable-format structure. Both ELO and Glicko-2 reach ~68% standalone prediction accuracy; combining them with light gradient boosting yields 70.2% accuracy and well-calibrated probabilities. The two rating systems' predictions account for 74.5% of the best model's predictive power, with auxiliary features contributing minimally — suggesting that long-run skill differentials are the dominant signal in pre-match snooker prediction.

The system is open-source and deployed as an interactive web application, supporting live player rankings, match prediction, head-to-head simulation, and tournament bracket Monte Carlo. The code is structured as a reusable Python package with 42 unit tests covering ELO updates, Glicko-2 against Glickman's published test vectors, calibration metrics, and feature pipelines.

---

## References

1. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4).
2. Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Pub.
3. Glickman, M. E. (1995). The Glicko system. https://www.glicko.net/glicko/glicko.pdf
4. Glickman, M. E. (2001). Dynamic paired comparison models with stochastic variances. *Journal of Applied Statistics*, 28(6).
5. Glickman, M. E. (2012). Example of the Glicko-2 system. http://www.glicko.net/glicko/glicko2.pdf
6. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15.
7. Kovalchik, S. (2016). Searching for the GOAT of tennis win prediction. *Journal of Quantitative Analysis in Sports*, 12(3).
8. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2.
9. Original Erdos Institute project: https://github.com/PubohH/2025-Summer-Erdos-Elo-Project
