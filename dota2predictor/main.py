"""
Точка входа.

  python main.py                 - прогон на синтетических данных (самопроверка)
  python main.py details.csv     - прогон на реальных данных OpenDota
                                   (сначала запустите data_loader.py)
"""

from __future__ import annotations

import sys

import pandas as pd

from features import build_features
from model import train_and_evaluate


def main() -> None:
    if len(sys.argv) > 1:
        print(f"Загружаю реальные данные: {sys.argv[1]}\n")
        df = pd.read_csv(sys.argv[1])
        df = df.dropna(subset=["radiant_win", "radiant_roster", "dire_roster"])
    else:
        from synthetic import make_dataset, theoretical_logloss

        print("Синтетический датасет (самопроверка пайплайна)\n")
        df = make_dataset()
        # Предел ниже которого log-loss опуститься не может в этом мире:
        tail = df.sort_values("start_time").tail(int(len(df) * 0.2))
        print(f"Теоретический предел log-loss (оракул): {theoretical_logloss(tail):.4f}\n")

    feats = build_features(df)
    print(f"Построено фичей: {len(feats)} матчей x {feats.shape[1]} колонок\n")
    train_and_evaluate(feats)


if __name__ == "__main__":
    main()
