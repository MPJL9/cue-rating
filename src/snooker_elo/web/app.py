"""FastAPI backend for snooker rating system."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from snooker_elo.web.engine import RatingEngine

engine: RatingEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize rating engines on startup."""
    global engine
    data_path = Path(__file__).parents[3] / "data" / "raw" / "matches.csv"
    engine = RatingEngine(str(data_path))
    engine.initialize()
    yield
    engine = None


app = FastAPI(
    title="Snooker Rating System API",
    description="ELO and Glicko-2 ratings for professional snooker players",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/ratings")
def get_ratings(
    system: str = Query("elo", pattern="^(elo|glicko2)$"),
    top: int = Query(50, ge=1, le=500),
):
    """Get top player ratings."""
    ratings = engine.get_ratings(system, top)
    return {"system": system, "players": ratings}


@app.get("/api/player/{name}")
def get_player(name: str):
    """Get detailed player profile."""
    player = engine.get_player(name)
    if player is None:
        return JSONResponse(status_code=404, content={"error": f"Player '{name}' not found"})
    return player


@app.post("/api/predict")
def predict_match(body: dict):
    """Predict match outcome between two players.

    Body: {"player1": "...", "player2": "...", "best_of": 9}
    """
    player1 = body.get("player1", "")
    player2 = body.get("player2", "")
    best_of = int(body.get("best_of", 9))

    if not player1 or not player2:
        return JSONResponse(status_code=400, content={"error": "player1 and player2 required"})

    prediction = engine.predict(player1, player2, best_of)
    if prediction is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Player(s) not found: {player1}, {player2}"},
        )
    return prediction


@app.get("/api/comparison")
def get_comparison():
    """Get ELO vs Glicko-2 comparison metrics."""
    return engine.get_comparison()


@app.get("/api/search")
def search_players(q: str = Query("", min_length=1)):
    """Search players by name prefix."""
    results = engine.search_players(q)
    return {"results": results[:20]}


@app.get("/api/stats")
def get_stats():
    """Get dataset statistics."""
    return engine.get_stats()


@app.get("/api/player/{name}/history")
def get_player_history(name: str):
    """Get rating history for a player (for charts)."""
    history = engine.get_rating_history(name)
    if history is None:
        return JSONResponse(status_code=404, content={"error": f"Player '{name}' not found"})
    return history


@app.post("/api/simulate")
def simulate_tournament(body: dict):
    """Monte Carlo tournament simulation.

    Body: {"players": ["Name1", "Name2", ...], "best_of": 9, "simulations": 10000}
    """
    players = body.get("players", [])
    best_of = int(body.get("best_of", 9))
    n_sims = min(int(body.get("simulations", 10000)), 50000)

    if len(players) < 2:
        return JSONResponse(status_code=400, content={"error": "Need at least 2 players"})

    result = engine.simulate_tournament(players, best_of, n_sims)
    return result


@app.get("/api/matches/recent")
def get_recent_matches(limit: int = Query(50, ge=1, le=200)):
    """Get most recent matches with predictions."""
    return engine.get_recent_matches(limit)


@app.get("/api/prime-times")
def get_prime_times(min_matches: int = Query(200, ge=50, le=1000)):
    """Get peak rating and prime years for experienced players."""
    return engine.get_prime_times(min_matches)
