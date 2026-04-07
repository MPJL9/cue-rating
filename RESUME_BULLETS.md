# Resume Bullets — Cue Rating

## Short Version (2-3 bullets)

- Built a full-stack snooker rating and match prediction system using ELO and Glicko-2 algorithms, achieving 70.2% prediction accuracy on 115,630 professional matches (1982–2025) — competitive with published academic benchmarks
- Optimized rating parameters via Maximum Likelihood Estimation (scipy) and reduced data pipeline runtime from hours to seconds through incremental computation (O(T²·M) → O(T·M))
- Deployed as a public web app (FastAPI + React) with player rankings, match predictor, Monte Carlo tournament simulator, and interactive methodology page with LaTeX equations

## Detailed Version (5-6 bullets)

- Designed and implemented ELO and Glicko-2 rating systems for professional snooker, processing 115,630 matches across 3,832 players in under 3 seconds using dict-based state optimization (50–100x speedup over pandas DataFrame approach)
- Optimized rating system parameters (K-factor, scaling divisor, volatility constraint) via Maximum Likelihood Estimation, improving prediction calibration over manually tuned baselines
- Built an incremental feature engineering pipeline that reduced data generation from O(T²·M) to O(T·M), cutting runtime from hours to seconds while adding novel features (head-to-head records, rating momentum, player inactivity)
- Trained and evaluated Logistic Regression, Random Forest, and Gradient Boosting models across 5 feature sets with temporal cross-validation, achieving 70.2% match prediction accuracy — within 2% of the estimated theoretical ceiling for pre-match snooker prediction
- Developed a full-stack web application (FastAPI + React + Recharts + KaTeX) featuring player rankings, match prediction, Monte Carlo tournament simulation, rating history charts, and a methodology page with rendered LaTeX equations
- Deployed publicly on Render with background initialization, precomputed API caches, and a single-container architecture serving both backend and frontend

## One-Liner (for project list)

Snooker rating system (ELO + Glicko-2) with ML match prediction (70.2% accuracy) and full-stack web app — Python, FastAPI, React, scikit-learn

## Links

- Live: https://cue-rating.onrender.com
- Code: https://github.com/MPJL9/cue-rating
