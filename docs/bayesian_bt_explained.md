# Bayesian Bradley-Terry for Snooker — Explained

This document teaches the math and intuition behind the Bayesian rating system added to this project. If you've worked with ELO before, this is a deeper, more principled way of thinking about the same problem.

---

## 1. The basic question

Given a bunch of match results, we want to assign each player a **skill number** such that:
- Higher skill = better player
- Differences in skill predict match outcomes

ELO does this with a simple update rule: after each match, nudge the winner up and the loser down by some amount based on how surprising the result was. Glicko-2 adds a second number per player (RD) to track uncertainty.

A **Bayesian** approach is fundamentally different. Instead of maintaining one (or two) numbers per player, we maintain a **full probability distribution** over each player's skill, and update those distributions using Bayes' rule whenever we see new data.

---

## 2. Bradley-Terry: the model

The Bradley-Terry model (Bradley & Terry, 1952) is a clean, classical way to relate latent skills to observed outcomes:

> If player *i* has skill *sᵢ* and player *j* has skill *sⱼ*, then the probability that *i* beats *j* in a single trial is:
>
> **P(i beats j) = sigmoid(sᵢ − sⱼ) = 1 / (1 + e^(sⱼ − sᵢ))**

Notice this is **exactly the same logistic function ELO uses** — but Bradley-Terry is the underlying probabilistic model that ELO is implicitly trying to fit. ELO is essentially online stochastic gradient descent on the Bradley-Terry log-likelihood. We're now going to fit Bradley-Terry properly using full Bayesian inference.

### Adapting to snooker

Snooker matches consist of multiple **frames**, and we already model frame-level outcomes (this is the whole point of our system). So instead of "player i beats player j" being a binary event, we observe:

> Player *i* won *fᵢ* frames out of *fᵢ + fⱼ* total frames against player *j*.

This is a **Binomial** observation:

> **fᵢ ~ Binomial(n, p)** where **n = fᵢ + fⱼ** and **p = sigmoid(sᵢ − sⱼ)**

Aggregating across many matches between the same pair of players, we sum the frame counts and the binomial assumption still holds (assuming frames are conditionally independent given skill, which is the same simplification ELO makes).

---

## 3. Bayesian inference: priors and posteriors

The Bayesian recipe has three ingredients:

### Prior: what we believe before seeing data

We don't know anything about a player's skill before observing any matches, so we use a **standard normal prior** on each player's skill:

> **sᵢ ~ Normal(0, 1)**

This says: "before any data, I think a typical player's skill is somewhere around 0, and most players are within ±2 of that." It's a vague but proper prior — it says "stay close to 0 unless the data strongly says otherwise."

### Likelihood: how the data depends on skills

This is the Bradley-Terry binomial we defined above:

> **fᵢⱼ ~ Binomial(nᵢⱼ, sigmoid(sᵢ − sⱼ))**

For all pairs (i, j) we've observed, we multiply these binomial likelihoods together. Each pair contributes one observation.

### Posterior: what we believe after seeing data

Bayes' rule combines them:

> **P(skills | data) ∝ P(data | skills) × P(skills)**
>
> = ∏ᵢⱼ Binomial(fᵢⱼ; nᵢⱼ, sigmoid(sᵢ − sⱼ)) × ∏ᵢ Normal(sᵢ; 0, 1)

The left side is the **posterior** — a joint probability distribution over the skills of all players, conditional on every match we've observed.

Why is this useful? Because the posterior contains *everything we know*. The mean tells you the best skill estimate. The variance tells you how confident we are. The shape can be skewed or multi-modal. We can compute things like *"what's the probability that Trump's skill is higher than Selby's?"* by integrating the posterior — something ELO simply cannot answer.

---

## 4. The catch: you can't compute this analytically

The expression above is a product of hundreds of binomials and normals over thousands of latent variables. There's no closed-form solution. We need a numerical method.

**MCMC (Markov Chain Monte Carlo)** is the standard tool. It doesn't compute the posterior directly — instead, it generates a long chain of samples drawn approximately from the posterior. With enough samples, you can compute any quantity (mean, variance, quantiles, probabilities) by averaging over the samples.

Modern MCMC for continuous parameters uses **Hamiltonian Monte Carlo (HMC)** with the **No-U-Turn Sampler (NUTS)** auto-tuning algorithm. PyMC implements this. In our model:

```python
with pm.Model() as model:
    skill = pm.Normal("skill", mu=0.0, sigma=1.0, shape=n_players)
    diff = skill[i_arr] - skill[j_arr]
    p = pm.math.sigmoid(diff)
    pm.Binomial("obs", n=n_total, p=p, observed=wins_i)
    trace = pm.sample(draws=1000, tune=500, chains=4)
```

What this does:
1. Defines `skill` as a vector of `n_players` random variables, each with a standard normal prior.
2. Computes the predicted frame win probability for each player pair via the Bradley-Terry formula.
3. Adds the observed frame counts as a Binomial likelihood.
4. Runs NUTS for 500 tuning steps (adaptive warm-up) + 1000 sampling steps × 4 parallel chains = **4000 posterior samples**.

The output is a `trace` object containing 4000 plausible joint configurations of all players' skills. We compute the per-player mean and standard deviation by averaging across samples.

---

## 5. What this gives you that ELO and Glicko-2 don't

### (a) Real uncertainty quantification

ELO gives you "Trump's skill is 1551." Glicko-2 gives you "1551 ± 48 (Gaussian)." The Bayesian model gives you **the actual posterior distribution** — which can be skewed, fat-tailed, or asymmetric. You can compute:

- "Probability that Trump is in the top 5" = fraction of posterior samples where Trump's skill rank is ≤ 5
- "95% credible interval for Trump's skill" = `[2.5th percentile, 97.5th percentile]`
- "Probability Trump beats Selby" = average of `sigmoid(Trump_skill − Selby_skill)` across all posterior samples

### (b) Posterior predictive checks

For any future match, we don't compute one win probability — we compute a **distribution** over win probabilities. If the posterior is concentrated, the prediction is confident. If the posterior is spread out (rare matchup, limited data), the prediction reflects that uncertainty.

### (c) Hierarchical extensions become natural

Once you've set up the model in PyMC, adding more structure is just a few lines:

- **Per-tournament random effects**: each tournament has its own offset (some players play better at the Crucible)
- **Time-varying skill**: model skill as a random walk over time, `sᵢ(t) = sᵢ(t-1) + ε`
- **Group-level priors**: rookies share a "rookie prior," veterans share a "veteran prior"

ELO and Glicko-2 cannot express any of this — they're stuck with their fixed update rules.

### (d) Honest about small samples

A player with 5 matches gets a wide posterior (still close to the prior). A player with 500 matches gets a narrow posterior. ELO and Glicko-2 do this approximately; Bayesian does it exactly.

---

## 6. Why we restrict to active players

The full dataset has **3,849 players** spanning 1982-2026. Fitting MCMC on all of them is:

- Computationally expensive (4000 samples × 3849 dimensions = 15.4M numbers, hours of MCMC)
- Statistically meaningless for players with very few matches (their posterior is essentially the prior)
- Confused by skill drift across decades — Stephen Hendry of 1995 is not the same player as Stephen Hendry of 2010

So we restrict to:
- Players with **≥100 matches in the last 2 years** (~150 active professionals)
- Aggregating frame counts per pair (instead of per-match), which is mathematically equivalent due to binomial sufficiency

This keeps MCMC fast (~1-2 minutes) and gives meaningful posteriors for the players who matter.

---

## 7. How we'll compare to ELO and Glicko-2

We'll evaluate all three systems on the same held-out test set with the same metrics:

| Metric | What it measures |
|--------|------------------|
| Accuracy | Did we predict the right winner? |
| Log loss | How confident were we in the right answer? |
| Brier score | Mean squared error of probability predictions |
| Calibration (ECE) | Do "70% probability" predictions actually win 70% of the time? |

The Bayesian model's prediction for a future match is computed by averaging over the posterior:

> P(i beats j in BO_n) ≈ (1/N) Σ_k P(i beats j in BO_n | s_i^(k), s_j^(k))

where `s^(k)` are the posterior samples. This is called **posterior predictive averaging** and properly accounts for skill uncertainty.

---

## 8. What the result will look like

After fitting, each player gets:

| Field | Description |
|-------|-------------|
| `skill_mean` | Posterior mean — best point estimate |
| `skill_std` | Posterior standard deviation — uncertainty |
| `skill_q025`, `skill_q975` | 95% credible interval |

We can report rankings as: *"Trump: skill 1.42 ± 0.08 (95% CI: 1.26 to 1.58)"* — meaning the data strongly suggests Trump's true skill is in that interval.

For a high-confidence player like Trump (1500+ matches), the posterior std is small. For a player with only a few recent matches, it's larger.

---

## 9. Why this matters for the resume / quant pitch

Quants and ML researchers care about probabilistic modeling because:

1. **Uncertainty matters in trading.** A trading signal with high confidence vs low confidence should be sized differently. Bayesian methods give you that natively.
2. **Calibration > accuracy.** Saying "70% probability" means nothing if you're systematically wrong. Calibration analysis (Section 8) is the test.
3. **Probabilistic programming is a real skill.** Knowing PyMC/Stan signals you can build custom models, not just plug into sklearn.
4. **Bradley-Terry generalizes.** The exact same model is used in: pairwise preference learning, recommender systems (LambdaMART), reinforcement learning from human feedback (RLHF), and rating systems for chess, Go, esports, and academic peer review.

When a quant interviewer sees "Bayesian Bradley-Terry model implemented in PyMC with full posterior calibration analysis," they think *"this person actually understands probability"* — which is the bar for the role.

---

## 10. References

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4).
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15.
- Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2.
- Glickman, M. E. (1999). Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 48(3) — derives Glicko as an approximation to a Bayesian model.
