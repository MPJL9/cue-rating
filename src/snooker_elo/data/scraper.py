"""Web scraper for snooker match data from cuetracker.net.

Ported from legacy scrape_one_tourn.py with improvements:
- Rate limiting (polite crawling)
- Retry with exponential backoff
- Incremental scraping (skip already-scraped tournaments)
- Data validation
- Direct output to matches.csv format
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None


def _check_deps():
    if requests is None:
        raise ImportError(
            "Scraping dependencies not installed. Run: pip install snooker-elo[scraping]"
        )


def scrape_tournament(url: str, delay: float = 1.5, max_retries: int = 3) -> pd.DataFrame:
    """Scrape match data from a single cuetracker.net tournament page.

    Args:
        url: Tournament URL (e.g. https://cuetracker.net/tournaments/.../2024/1234).
        delay: Seconds to wait before making request (rate limiting).
        max_retries: Number of retry attempts on failure.

    Returns:
        DataFrame with columns [player1, player2, score1, score2, best_of,
        tournament_id, date, year].
    """
    _check_deps()

    time.sleep(delay)

    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    soup = BeautifulSoup(response.text, "html.parser")
    all_matches = soup.find_all("div", class_="match")

    # Extract tournament metadata from URL
    url_parts = url.rstrip("/").split("/")
    tournament_id = url_parts[-1]
    year = url_parts[-2]

    rows = []
    for match in all_matches:
        try:
            # Player names
            names = match.find_all("div", class_="matchResultText")
            if len(names) < 2:
                continue
            player1 = names[0].text.strip()
            player2 = names[1].text.strip()

            # Scores
            scores = match.find_all("span", class_="matchResultText")
            if len(scores) < 2:
                continue
            score1 = int(scores[0].text.strip())
            score2 = int(scores[1].text.strip())

            # Skip walkovers
            if score1 == 0 and score2 == 0:
                continue

            # Best of
            best_of_elem = match.find("span", class_="best_of")
            best_of = int(best_of_elem.text.strip("()")) if best_of_elem else 0

            # Date
            date_elem = match.find("div", class_="col-12 played_on")
            date = date_elem.text.strip() if date_elem else ""

            # Ensure player1 is the winner (score1 >= score2)
            if score1 < score2:
                player1, player2 = player2, player1
                score1, score2 = score2, score1

            rows.append({
                "player1": player1,
                "player2": player2,
                "score1": score1,
                "score2": score2,
                "best_of": best_of,
                "tournament_id": tournament_id,
                "date": date,
                "year": int(year),
            })
        except (ValueError, IndexError, AttributeError):
            continue  # Skip malformed matches

    df = pd.DataFrame(rows)

    # Reverse order to get chronological (earliest round first)
    if len(df) > 0:
        df = df.iloc[::-1].reset_index(drop=True)

    return df


def scrape_tournament_list(year: int, delay: float = 1.5) -> list[dict]:
    """Scrape the list of tournaments for a given year from cuetracker.net.

    Returns:
        List of dicts with keys [name, url, tournament_id].
    """
    _check_deps()

    time.sleep(delay)
    url = f"https://cuetracker.net/tournaments/snooker/{year}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tournaments = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if f"/tournaments/snooker/{year}/" in href and href.count("/") >= 4:
            tid = href.rstrip("/").split("/")[-1]
            name = link.text.strip()
            if name and tid.isdigit():
                tournaments.append({
                    "name": name,
                    "url": f"https://cuetracker.net{href}" if href.startswith("/") else href,
                    "tournament_id": tid,
                })

    # Deduplicate
    seen = set()
    unique = []
    for t in tournaments:
        if t["tournament_id"] not in seen:
            seen.add(t["tournament_id"])
            unique.append(t)

    return unique


def scrape_new_matches(
    existing_path: str | Path,
    years: list[int] | None = None,
    delay: float = 1.5,
) -> pd.DataFrame:
    """Incrementally scrape new tournaments not in existing dataset.

    Args:
        existing_path: Path to existing matches.csv.
        years: Years to scrape. Defaults to [2025, 2026].
        delay: Rate limiting delay between requests.

    Returns:
        DataFrame of newly scraped matches (not yet appended to file).
    """
    _check_deps()

    if years is None:
        years = [2025, 2026]

    # Load existing tournament IDs
    existing = pd.read_csv(existing_path, dtype={"tournament_id": str})
    existing_ids = set(existing["tournament_id"].unique())
    print(f"Existing dataset: {len(existing)} matches, {len(existing_ids)} tournaments")

    all_new = []
    for year in years:
        print(f"\nScraping tournament list for {year}...")
        try:
            tournaments = scrape_tournament_list(year, delay=delay)
        except Exception as e:
            print(f"  Failed to get tournament list: {e}")
            continue

        new_tournaments = [t for t in tournaments if t["tournament_id"] not in existing_ids]
        print(f"  Found {len(tournaments)} tournaments, {len(new_tournaments)} new")

        for t in new_tournaments:
            print(f"  Scraping: {t['name']} (ID: {t['tournament_id']})...")
            try:
                df = scrape_tournament(t["url"], delay=delay)
                if len(df) > 0:
                    all_new.append(df)
                    print(f"    -> {len(df)} matches")
                else:
                    print(f"    -> No matches found")
            except Exception as e:
                print(f"    -> Error: {e}")

    if all_new:
        result = pd.concat(all_new, ignore_index=True)
        print(f"\nTotal new matches: {len(result)}")
        return result
    else:
        print("\nNo new matches found")
        return pd.DataFrame(
            columns=["player1", "player2", "score1", "score2", "best_of",
                     "tournament_id", "date", "year"]
        )
