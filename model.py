"""
Обучение и оценка модели.

Три уровня сравнения (это важно для честной самооценки модели):
  1. Baseline "всегда 50%" / "всегда radiant"
  2. Чистый Glicko-2 (glicko_prob_a как прогноз)
  3. LightGBM поверх всех фичей + изотоническая калибровка

Валидация СТРОГО по времени: train - прошлое, test - будущее.
Метрики: log-loss и Brier (качество вероятностей), accuracy - справочно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from features import FEATURE_COLS


def time_split(feats: pd.DataFrame, test_frac: float = 0.2):
    """Хронологический split: последние test_frac матчей - в тест."""
    feats = feats.sort_values("start_time").reset_index(drop=True)
    cut = int(len(feats) * (1 - test_frac))
    return feats.iloc[:cut], feats.iloc[cut:]


def evaluate(y_true: np.ndarray, p: np.ndarray, name: str) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    res = {
        "model": name,
        "log_loss": log_loss(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "accuracy": accuracy_score(y_true, p >= 0.5),
    }
    print(
        f"{name:<22} log_loss={res['log_loss']:.4f}  "
        f"brier={res['brier']:.4f}  acc={res['accuracy']:.3f}"
    )
    return res


def train_and_evaluate(feats: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    train, test = time_split(feats)
    x_tr, y_tr = train[FEATURE_COLS], train["target"].values
    x_te, y_te = test[FEATURE_COLS], test["target"].values

    print(f"train: {len(train)} matches, test: {len(test)} matches\n")
    results = []

    # 1. Baselines
    results.append(evaluate(y_te, np.full(len(y_te), 0.5), "baseline 50/50"))
    results.append(
        evaluate(y_te, np.full(len(y_te), y_tr.mean()), "baseline base-rate")
    )

    # 2. Чистый Glicko-2
    results.append(evaluate(y_te, test["glicko_prob_a"].values, "glicko2 only"))

    # 3. LightGBM + калибровка.
    # Датасеты про-сцены небольшие (тысячи матчей), поэтому модель
    # намеренно неглубокая и с сильной регуляризацией.
    base = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=42,
        verbose=-1,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(x_tr, y_tr)
    p = model.predict_proba(x_te)[:, 1]
    results.append(evaluate(y_te, p, "lgbm + isotonic"))

    # Важность фичей (из откалиброванных подмоделей)
    imps = np.mean(
        [c.estimator.feature_importances_ for c in model.calibrated_classifiers_],
        axis=0,
    )
    imp_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": imps})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\nFeature importance:")
    print(imp_df.to_string(index=False))

    return model, pd.DataFrame(results)


def predict_match(model, pool, roster_a, roster_b, extra: dict) -> float:
    """
    Прогноз для будущего матча. extra - словарь с остальными фичами
    (форма, h2h, отдых...), собранными тем же кодом, что и в features.py.
    """
    row = {
        "glicko_prob_a": pool.predict(roster_a, roster_b),
        "rating_diff": pool.team_rating(roster_a).rating
        - pool.team_rating(roster_b).rating,
        "rd_a": pool.team_rating(roster_a).rd,
        "rd_b": pool.team_rating(roster_b).rd,
        **extra,
    }
    x = pd.DataFrame([row])[FEATURE_COLS]
    return float(model.predict_proba(x)[0, 1])
