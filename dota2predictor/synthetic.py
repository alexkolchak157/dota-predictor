"""
Синтетический датасет про-матчей для проверки пайплайна без интернета.

Устройство мира: у каждого игрока есть скрытая 'истинная сила',
которая медленно дрейфует; команды - составы из 5 игроков с редкими
заменами (стенд-ины); исход карты - Бернулли от разницы сил составов
+ бонус стороны radiant. Если модель улавливает скрытые силы,
её log-loss должен приближаться к теоретическому пределу мира.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)


def make_dataset(
    n_teams: int = 40,
    n_matches: int = 6000,
    skill_sd: float = 1.0,
    drift_sd: float = 0.02,
    standin_prob: float = 0.05,
    radiant_bonus: float = 0.15,
) -> pd.DataFrame:
    n_players = n_teams * 5 + 50  # 50 свободных агентов на замены
    skills = RNG.normal(0, skill_sd, n_players)
    rosters = {t: list(range(t * 5, t * 5 + 5)) for t in range(n_teams)}
    free_agents = list(range(n_teams * 5, n_players))

    rows = []
    ts = 1_700_000_000.0
    for mid in range(n_matches):
        ts += float(RNG.exponential(3600 * 4))
        skills += RNG.normal(0, drift_sd, n_players)  # дрейф формы

        ta, tb = RNG.choice(n_teams, size=2, replace=False)
        roster_a = list(rosters[ta])
        roster_b = list(rosters[tb])
        # редкие стенд-ины
        if RNG.random() < standin_prob:
            roster_a[RNG.integers(5)] = int(RNG.choice(free_agents))
        if RNG.random() < standin_prob:
            roster_b[RNG.integers(5)] = int(RNG.choice(free_agents))

        diff = skills[roster_a].mean() - skills[roster_b].mean() + radiant_bonus
        p_a = 1.0 / (1.0 + np.exp(-1.6 * diff))
        a_won = RNG.random() < p_a

        rows.append(
            {
                "match_id": mid,
                "start_time": ts,
                "radiant_win": bool(a_won),
                "radiant_team_id": int(ta),
                "dire_team_id": int(tb),
                "radiant_roster": ";".join(map(str, roster_a)),
                "dire_roster": ";".join(map(str, roster_b)),
                "true_p_a": p_a,  # для расчёта теоретического предела
            }
        )
    return pd.DataFrame(rows)


def theoretical_logloss(df: pd.DataFrame) -> float:
    """Log-loss всезнающего оракула, знающего истинные вероятности мира."""
    p = df["true_p_a"].to_numpy()
    y = df["radiant_win"].to_numpy().astype(float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
