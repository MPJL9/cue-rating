"""FastAPI backend for snooker rating system."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from snooker_elo.web.engine import RatingEngine

engine: RatingEngine | None = None
engine_ready = threading.Event()


def _find_data_dir() -> Path:
    """Find the data/raw/ directory, works both locally and on Render."""
    candidates = [
        Path(__file__).parents[3] / "data" / "raw",  # Local dev
        Path.cwd() / "data" / "raw",                 # Render: cwd is repo root
        Path("/opt/render/project/src/data/raw"),     # Render explicit
    ]
    for p in candidates:
        if (p / "matches.csv").exists():
            return p
    raise FileNotFoundError(f"Cannot find data/raw/matches.csv. Tried: {candidates}")


def _init_engine():
    """Initialize engine in background thread."""
    global engine
    data_dir = _find_data_dir()
    engine = RatingEngine(str(data_dir / "matches.csv"))
    engine.initialize()
    engine_ready.set()
    print("Engine ready — all endpoints active")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start server immediately, compute ratings in background."""
    thread = threading.Thread(target=_init_engine, daemon=True)
    thread.start()
    yield


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


def _check_ready():
    """Return error response if engine not ready yet."""
    if not engine_ready.is_set():
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "message": "Computing ratings... Please wait ~30s and refresh.",
            },
        )
    return None


@app.get("/api/status")
def get_status():
    """Check if engine is ready."""
    if engine_ready.is_set():
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "loading"})


@app.get("/api/ratings")
def get_ratings(
    system: str = Query("elo", pattern="^(elo|glicko2)$"),
    top: int = Query(50, ge=1, le=500),
):
    """Get top player ratings."""
    err = _check_ready()
    if err:
        return err
    ratings = engine.get_ratings(system, top)
    return {"system": system, "players": ratings}


@app.get("/api/player/{name}")
def get_player(name: str):
    """Get detailed player profile."""
    err = _check_ready()
    if err:
        return err
    player = engine.get_player(name)
    if player is None:
        return JSONResponse(status_code=404, content={"error": f"Player '{name}' not found"})
    return player


@app.post("/api/predict")
def predict_match(body: dict):
    """Predict match outcome between two players.

    Body: {"player1": "...", "player2": "...", "best_of": 9}
    """
    err = _check_ready()
    if err:
        return err
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
    err = _check_ready()
    if err:
        return err
    return engine.get_comparison()


@app.get("/api/search")
def search_players(q: str = Query("", min_length=1)):
    """Search players by name prefix."""
    err = _check_ready()
    if err:
        return err
    results = engine.search_players(q)
    return {"results": results[:20]}


@app.get("/api/stats")
def get_stats():
    """Get dataset statistics."""
    err = _check_ready()
    if err:
        return err
    return engine.get_stats()


@app.get("/api/player/{name}/history")
def get_player_history(name: str):
    """Get rating history for a player (for charts)."""
    err = _check_ready()
    if err:
        return err
    history = engine.get_rating_history(name)
    if history is None:
        return JSONResponse(status_code=404, content={"error": f"Player '{name}' not found"})
    return history


@app.post("/api/simulate")
def simulate_tournament(body: dict):
    """Monte Carlo tournament simulation.

    Body: {"players": ["Name1", "Name2", ...], "best_of": 9, "simulations": 10000}
    """
    err = _check_ready()
    if err:
        return err
    players = body.get("players", [])
    best_of = int(body.get("best_of", 9))
    n_sims = min(int(body.get("simulations", 10000)), 50000)

    if len(players) < 2:
        return JSONResponse(status_code=400, content={"error": "Need at least 2 players"})

    fmt = body.get("format_type", "uniform")
    result = engine.simulate_tournament(players, best_of, n_sims, fmt)
    return result


@app.post("/api/simulate/bracket")
def simulate_bracket(body: dict):
    """Simulate one tournament bracket with scores.

    Body: {"players": [...], "best_of": 9, "format_type": "world_championship"}
    """
    err = _check_ready()
    if err:
        return err
    players = body.get("players", [])
    best_of = int(body.get("best_of", 9))
    fmt = body.get("format_type", "uniform")
    if len(players) < 2:
        return JSONResponse(
            status_code=400, content={"error": "Need at least 2 players"}
        )
    return engine.simulate_single_bracket(players, best_of, fmt)


@app.get("/api/matches/recent")
def get_recent_matches(limit: int = Query(30, ge=1, le=100)):
    """Get recent tournaments with matches and predictions."""
    err = _check_ready()
    if err:
        return err
    return engine.get_recent_matches(limit)


@app.get("/api/prime-times")
def get_prime_times(min_matches: int = Query(200, ge=50, le=1000)):
    """Get peak rating and prime years for experienced players."""
    err = _check_ready()
    if err:
        return err
    return engine.get_prime_times(min_matches)


# ── Serve React frontend (built files) ──

def _find_frontend_dist() -> Path | None:
    """Find the built frontend dist/ directory."""
    candidates = [
        Path(__file__).parents[3] / "frontend" / "dist",
        Path.cwd() / "frontend" / "dist",
        Path("/opt/render/project/src/frontend/dist"),
    ]
    for p in candidates:
        if p.is_dir() and (p / "index.html").exists():
            return p
    return None


_dist = _find_frontend_dist()
if _dist:
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(_dist / "assets")), name="assets")

    # Catch-all: serve index.html for any non-API route (SPA routing)
    @app.get("/{path:path}")
    def serve_spa(path: str):
        file_path = _dist / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_dist / "index.html"))
