"""
Загрузка про-матчей Dota 2 из OpenDota API.

Лимиты бесплатного тарифа: 60 запросов/мин И 2000 запросов/день.
Для ~3000 матчей нужно либо 2 дня (скрипт умеет докачивать с места
остановки), либо бесплатный API-ключ с opendota.com (лимит выше).

При 429 (Too Many Requests) скрипт НЕ пропускает матч, а ждёт и
повторяет попытку. Прогресс сохраняется каждые 25 матчей - запуск
можно прерывать и возобновлять сколько угодно раз.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY")  # export OPENDOTA_API_KEY=...
SLEEP = 1.2 if API_KEY else 6.0  # без ключа: ~600 матчей/час, зато без 429
MAX_RETRIES = 6


def _get(url: str, params: dict | None = None) -> requests.Response | None:
    """GET с повторами и экспоненциальным backoff на 429/5xx."""
    params = dict(params or {})
    if API_KEY:
        params["api_key"] = API_KEY

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            wait = 2**attempt * 5
            print(f"  network error ({exc}), retry in {wait}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            # Сервер может подсказать, сколько ждать
            wait = int(resp.headers.get("Retry-After", 0)) or 2**attempt * 15
            print(f"  429 rate limit, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            wait = 2**attempt * 10
            print(f"  {resp.status_code} server error, retry in {wait}s")
            time.sleep(wait)
            continue
        print(f"  {resp.status_code} for {url}, skipping")
        return None

    print(f"  giving up after {MAX_RETRIES} retries: {url}")
    print("  Похоже, исчерпана ДНЕВНАЯ квота (2000 req). Продолжите завтра -")
    print("  скрипт докачает с места остановки. Или заведите API-ключ.")
    return None


def fetch_pro_matches(n_pages: int = 30, out: str = "matches.csv") -> pd.DataFrame:
    """Список последних про-матчей (n_pages * 100). Докачивает при повторном запуске."""
    out_path = Path(out)
    if out_path.exists():
        print(f"{out} уже существует - использую его (удалите файл для перекачки)")
        return pd.read_csv(out_path)

    rows: list[dict] = []
    less_than: int | None = None
    for page in range(n_pages):
        params = {"less_than_match_id": less_than} if less_than else {}
        resp = _get(f"{API}/proMatches", params)
        if resp is None:
            break
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        less_than = batch[-1]["match_id"]
        print(f"page {page + 1}/{n_pages}: total {len(rows)} matches")
        time.sleep(SLEEP)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def fetch_match_details(match_ids: list[int], out: str = "details.csv") -> pd.DataFrame:
    """Детали матчей с СОСТАВАМИ. Возобновляемая: уже скачанное не качает снова."""
    out_path = Path(out)
    rows: list[dict] = []
    done: set[int] = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = set(prev["match_id"].astype(int))
        print(f"resume: {len(done)} матчей уже скачано, осталось {len(set(match_ids) - done)}")

    todo = [m for m in match_ids if m not in done]
    for i, mid in enumerate(todo):
        resp = _get(f"{API}/matches/{mid}")
        if resp is None:
            break  # квота кончилась - сохраняемся и выходим, не теряя прогресс
        m = resp.json()
        players = m.get("players", [])
        radiant = [p.get("account_id") for p in players if p.get("isRadiant")]
        dire = [p.get("account_id") for p in players if not p.get("isRadiant")]
        rows.append(
            {
                "match_id": mid,
                "start_time": m.get("start_time"),
                "duration": m.get("duration"),
                "radiant_win": m.get("radiant_win"),
                "radiant_team_id": (m.get("radiant_team") or {}).get("team_id"),
                "dire_team_id": (m.get("dire_team") or {}).get("team_id"),
                "leagueid": m.get("leagueid"),
                "patch": m.get("patch"),
                "radiant_roster": ";".join(str(x) for x in radiant),
                "dire_roster": ";".join(str(x) for x in dire),
            }
        )
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"{len(rows)} matches saved (checkpoint)")
        time.sleep(SLEEP)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"итого в {out}: {len(df)} матчей")
    return df


if __name__ == "__main__":
    matches = fetch_pro_matches(n_pages=30)
    ids = matches["match_id"].dropna().astype(int).tolist()
    fetch_match_details(ids)
