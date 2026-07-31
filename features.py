"""
Построение фичей для модели. Ключевой принцип - НИКАКОГО заглядывания
в будущее: матчи обрабатываются строго в хронологическом порядке,
фичи матча считаются ДО обновления рейтингов его результатом.
"""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from glicko2 import PlayerPool

FORM_WINDOW = 10  # последних карт для расчёта формы


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вход: DataFrame с колонками
      match_id, start_time, radiant_win (bool),
      radiant_roster, dire_roster ('id1;id2;id3;id4;id5'),
      radiant_team_id, dire_team_id, patch (опц.)
    Выход: DataFrame фичей + target, по одному ряду на матч.
    """
    df = df.sort_values("start_time").reset_index(drop=True)

    pool = PlayerPool()
    team_form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    team_last_ts: dict[str, float] = {}
    h2h: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # wins_a, wins_b

    rows: list[dict] = []
    for m in df.itertuples(index=False):
        roster_a = str(m.radiant_roster).split(";")
        roster_b = str(m.dire_roster).split(";")
        if len(roster_a) != 5 or len(roster_b) != 5:
            continue
        ta, tb = str(m.radiant_team_id), str(m.dire_team_id)
        ts = float(m.start_time)

        # --- фичи ДО матча ---
        ra = pool.team_rating(roster_a)
        rb = pool.team_rating(roster_b)
        key = (min(ta, tb), max(ta, tb))
        wins = h2h[key]
        a_first = ta <= tb
        h2h_a = wins[0] if a_first else wins[1]
        h2h_b = wins[1] if a_first else wins[0]
        h2h_total = h2h_a + h2h_b

        form_a = team_form[ta]
        form_b = team_form[tb]
        exp_players_a = sum(pool.get(p).n_games for p in roster_a) / 5
        exp_players_b = sum(pool.get(p).n_games for p in roster_b) / 5

        rows.append(
            {
                "match_id": m.match_id,
                "start_time": ts,
                # Glicko-2
                "glicko_prob_a": pool.predict(roster_a, roster_b),
                "rating_diff": ra.rating - rb.rating,
                "rd_a": ra.rd,
                "rd_b": rb.rd,
                # форма
                "form_a": sum(form_a) / len(form_a) if form_a else 0.5,
                "form_b": sum(form_b) / len(form_b) if form_b else 0.5,
                "form_diff": (sum(form_a) / len(form_a) if form_a else 0.5)
                - (sum(form_b) / len(form_b) if form_b else 0.5),
                # H2H (сглаженный, чтобы 1-0 не значил "100%")
                "h2h_rate_a": (h2h_a + 1) / (h2h_total + 2),
                "h2h_n": h2h_total,
                # усталость/простой (дни с прошлой игры)
                "rest_days_a": (ts - team_last_ts.get(ta, ts)) / 86400,
                "rest_days_b": (ts - team_last_ts.get(tb, ts)) / 86400,
                # опыт составов в датасете
                "exp_a": exp_players_a,
                "exp_b": exp_players_b,
                # сторона: radiant исторически выигрывает ~52-53% карт
                "is_radiant_a": 1,
                "target": int(bool(m.radiant_win)),
            }
        )

        # --- обновление состояния ПОСЛЕ фиксации фичей ---
        a_won = bool(m.radiant_win)
        pool.record_match(roster_a, roster_b, a_won, ts)
        team_form[ta].append(1 if a_won else 0)
        team_form[tb].append(0 if a_won else 1)
        team_last_ts[ta] = ts
        team_last_ts[tb] = ts
        if a_first:
            wins[0 if a_won else 1] += 1
        else:
            wins[1 if a_won else 0] += 1

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "glicko_prob_a",
    "rating_diff",
    "rd_a",
    "rd_b",
    "form_a",
    "form_b",
    "form_diff",
    "h2h_rate_a",
    "h2h_n",
    "rest_days_a",
    "rest_days_b",
    "exp_a",
    "exp_b",
]
