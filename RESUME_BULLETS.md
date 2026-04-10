# Resume Bullets — Cue Rating

## Quant Researcher Version (3-4 bullets)

- Implemented and compared three rating systems for professional snooker — frame-weighted ELO, Glicko-2, and a fully Bayesian Bradley-Terry model fit via NUTS in PyMC — on 117,530 matches (1982–2026), with parameters tuned by Maximum Likelihood Estimation rather than grid search
- Built a calibration analysis framework (log loss, Brier score, expected calibration error, reliability diagrams) demonstrating that the best model achieves ECE = 0.011 — well-calibrated probabilistic predictions, not just raw accuracy
- Benchmarked against the official World Snooker Tour rankings as a free public proxy for bookmaker odds: outperforms by **+5.2 percentage points in accuracy and 12% lower log loss** on 4,777 held-out matches
- Optimized the data pipeline from O(T²·M) to O(T·M) via incremental computation; the production API serves predictions in <1ms after a 0.12s precomputed-cache load

## ML Engineer Version (3-4 bullets)

- Built a full-stack snooker rating and match prediction system from data acquisition through deployment: scraped 117K matches, implemented ELO, Glicko-2, and Bayesian Bradley-Terry rating systems, trained 3 ML classifiers, achieved 70.2% match prediction accuracy on a held-out test set
- Optimized the rating engine with dict-based player state, achieving a 50–100× speedup over the original pandas DataFrame approach (117K matches processed in 0.3s for ELO, 2.1s for Glicko-2)
- Developed a FastAPI + React + Recharts web application with 7 pages (rankings, predictions, Monte Carlo tournament bracket simulation, player profiles with rating-history charts, methodology with LaTeX equations); deployed on Render with single-container architecture serving both API and SPA
- Wrote 42 unit tests covering rating math, calibration metrics, and feature pipelines; verified Glicko-2 against Glickman's published test vectors

## Detailed Version (5-6 bullets, mix-and-match)

- **Three rating systems implemented from first principles**: frame-weighted ELO (chess-style logistic with binomial-derived match probabilities), Glicko-2 (Glickman's full algorithm with Illinois-method volatility update), and a Bayesian Bradley-Terry model in PyMC sampled with NUTS (4 chains × 2000 draws). Bayesian model uses posterior predictive averaging for proper uncertainty propagation
- **Principled parameter optimization via MLE**: maximized log-likelihood of observed frame outcomes using `scipy.optimize.minimize`, replacing the original project's hand-picked K-factor with optimal values (K=9.77, scaling divisor=327.15)
- **Calibration over accuracy**: best model achieves Expected Calibration Error of 0.011 — predictions of "70%" actually win 70% of the time. Verified via 10-bin reliability diagrams against three baseline systems
- **Beat the public baseline**: outperforms the official World Snooker Tour rankings by 5.2 percentage points in accuracy and reduces log loss by 12% on 4,777 matches from 2015–2019
- **Algorithmic optimization**: reduced data pipeline runtime from O(T²·M) to O(T·M) through incremental feature generation, processing 117,530 matches in seconds. Added head-to-head records, rating momentum, and inactivity features
- **Production deployment**: full-stack web app (FastAPI + React + Recharts + KaTeX) on Render free tier, with precomputed pickle cache loading 117K matches in 0.12 seconds. Single-container architecture serves both backend and SPA

## One-Liner (project list)

Snooker rating systems (ELO, Glicko-2, Bayesian Bradley-Terry) with MLE-optimized parameters, calibration analysis, and full-stack web app — Python, PyMC, FastAPI, React. Beats official rankings by 5.2pp accuracy and 12% log loss.

## Links

- **Live demo:** https://cue-rating.onrender.com
- **Code:** https://github.com/MPJL9/cue-rating
- **Technical writeup:** [docs/writeup.md](docs/writeup.md)
