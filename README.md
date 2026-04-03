# Snooker Rating System

ELO and Glicko-2 rating systems for professional snooker, with ML match prediction models and a web application.

> This project grew out of [2025-Summer-Erdos-Elo-Project](https://github.com/PubohH/2025-Summer-Erdos-Elo-Project), a summer 2025 project at the Erdos Institute. The original project applied a basic ELO system to snooker. This repo is a complete rewrite with Glicko-2, MLE-optimized parameters, an incremental pipeline, and a full-stack web app.

## Results

**115,630 matches** from 1982-2025 across **1,085 tournaments** and **3,832 players**.

### Rating System Comparison (34,521 matches, last 300 tournaments)

| Metric | ELO | Glicko-2 | Winner |
|--------|-----|----------|--------|
| Accuracy | 69.1% | 68.8% | ELO |
| Log Loss | 0.586 | 0.594 | ELO |
| Calibration (ECE) | 0.394 | 0.382 | Glicko-2 |
| Frame MAE | 0.218 | 0.214 | Glicko-2 |

### ML Model Performance

| Model | Features | Accuracy |
|-------|----------|----------|
| Gradient Boosting | All (ELO + Glicko-2 + stats) | **70.1%** |
| Logistic Regression | All features | 70.0% |
| Gradient Boosting | Combined ratings only | 69.9% |
| Random Forest | Glicko-2 only (4 features) | 69.5% |
| Pure ELO baseline | — | 68.5% |

### MLE-Optimized Parameters

| System | Parameter | Default | Optimized |
|--------|-----------|---------|-----------|
| ELO | K-factor | 8.0 | 9.77 |
| ELO | Divisor | 400 | 327.15 |
| Glicko-2 | Tau | 0.5 | 1.488 |

## Architecture

```
snooker_elo/
├── src/snooker_elo/
│   ├── ratings/          # ELO + Glicko-2 implementations
│   │   ├── base.py       # RatingSystem ABC, shared interface
│   │   ├── elo.py        # Frame-weighted ELO (0.34s for 115K matches)
│   │   ├── glicko2.py    # Full Glickman algorithm with RD + volatility
│   │   └── optimization.py  # MLE parameter optimization
│   ├── data/
│   │   ├── loader.py     # Data loading + validation
│   │   └── scraper.py    # cuetracker.net scraper with rate limiting
│   ├── features/
│   │   └── generator.py  # O(T·M) incremental feature pipeline
│   ├── models/
│   │   └── classification.py  # LR, RF, Gradient Boosting
│   ├── evaluation/
│   │   ├── metrics.py    # Log-loss, Brier, calibration, ECE
│   │   └── comparison.py # ELO vs Glicko-2 comparison framework
│   └── web/
│       ├── app.py        # FastAPI backend
│       └── engine.py     # Precomputed rating engine
├── frontend/             # React SPA (Vite + Recharts)
├── tests/                # 42 tests (pytest)
├── data/raw/             # matches.csv (115K matches)
└── legacy/               # Original Erdos Institute project
```

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest -v
```

### Start the API

```bash
pip install -e ".[web]"
uvicorn snooker_elo.web.app:app --reload
```

API docs at http://localhost:8000/docs

### Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

### Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ratings?system=elo&top=50` | Player rankings |
| GET | `/api/player/{name}` | Player profile + recent matches |
| POST | `/api/predict` | Match prediction (body: `{player1, player2, best_of}`) |
| GET | `/api/comparison` | ELO vs Glicko-2 comparison |
| GET | `/api/search?q=Trump` | Player search |
| GET | `/api/stats` | Dataset statistics |

## How It Works

### ELO Rating (adapted for snooker)

Unlike chess ELO, this system rates **frame win probability**, not match win probability:

- **Expected frame win rate**: `E = 1 / (1 + exp((R2 - R1) / 327))`
- **Update**: `R_new = R + K × total_frames × (actual_rate - expected_rate)`
- **Match win probability**: Derived via binomial formula from frame probability

The frame-based approach enables predicting exact match scores (e.g., P(5-3) in best-of-9).

### Glicko-2

Extends ELO with two additional parameters per player:

- **Rating Deviation (RD)**: Uncertainty in the rating. High RD = less certain (new/inactive players).
- **Volatility**: How consistent a player's performance is.

Players returning from breaks (e.g., Zhao Xintong after 2023-2024 ban) automatically get higher RD, making predictions less confident — something ELO cannot express.

### Parameter Optimization

Parameters are chosen via **Maximum Likelihood Estimation**, not manual tuning:

```
LL = Σ [frames_won_i × log(p_i) + frames_lost_i × log(1 - p_i)]
```

Maximized using `scipy.optimize.minimize` over the full match history.

## Data

- **1982-2020**: Kaggle dataset (cuetracker.net origin)
- **2020-2025**: Scraped from cuetracker.net
- **Format**: `player1, player2, score1, score2, best_of, tournament_id, date, year`
- Player1 is always the winner (`score1 >= score2`)

### Scrape New Data

```bash
pip install -e ".[scraping]"
python -c "
from snooker_elo.data.scraper import scrape_new_matches
new = scrape_new_matches('data/raw/matches.csv', years=[2025, 2026])
"
```

## Performance

| Operation | Time |
|-----------|------|
| ELO: process 115K matches | 0.34s |
| Glicko-2: process 115K matches | 2.1s |
| Feature generation (300 tournaments) | 5.7s |
| Full API startup | 2.6s |
| MLE optimization (ELO) | ~135s |

## Testing

```
42 tests covering:
- ELO update formula, known value regression
- Binomial match probability (all formats)
- Glicko-2 core math, RD inflation, paper test vectors
- Feature generation: no data leakage, swap correctness
- Real data integration tests
```
