"""
Glicko-2 rating system (Glickman, 2013) для киберспорта.

Отличие от Elo: каждый игрок имеет не только рейтинг (mu), но и
неопределённость рейтинга (RD/phi) и волатильность (sigma).
Новый игрок или игрок после долгого перерыва имеет высокий RD ->
его матчи сильнее двигают рейтинг, а прогнозы по нему менее уверенные.

Реализация работает в "внутренней" шкале Glicko-2 (mu, phi),
конвертация в привычную шкалу (1500 +/- RD) - через свойства.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Константы конвертации между шкалой Glicko-1 (1500/350) и Glicko-2
_SCALE = 173.7178
_BASE = 1500.0

# Ограничитель волатильности. tau из [0.3, 1.2]; для киберспорта,
# где сила команд меняется быстро (патчи, трансферы), берём выше среднего.
DEFAULT_TAU = 0.7
CONVERGENCE_EPS = 1e-6


@dataclass
class Rating:
    """Рейтинг одного игрока/команды."""

    mu: float = 0.0          # рейтинг во внутренней шкале (0 == 1500)
    phi: float = 350.0 / _SCALE  # неопределённость (RD 350 для новичка)
    sigma: float = 0.06      # волатильность
    n_games: int = 0
    last_ts: float | None = None  # unix time последней игры (для decay)

    @property
    def rating(self) -> float:
        """Рейтинг в привычной шкале (новичок = 1500)."""
        return _BASE + self.mu * _SCALE

    @property
    def rd(self) -> float:
        """Rating deviation в привычной шкале."""
        return self.phi * _SCALE

    def pre_rating_rd(self, rating_periods_inactive: float = 0.0) -> float:
        """phi с учётом простоя: неопределённость растёт со временем."""
        return min(
            math.sqrt(self.phi**2 + (rating_periods_inactive + 1) * self.sigma**2),
            350.0 / _SCALE,
        )


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def expected_score(r: Rating, opp: Rating) -> float:
    """P(победа r над opp) по текущим рейтингам."""
    return 1.0 / (1.0 + math.exp(-_g(math.hypot(r.phi, opp.phi)) * (r.mu - opp.mu)))


def _new_sigma(r: Rating, delta: float, v: float, tau: float) -> float:
    """Итерационный поиск новой волатильности (Illinois algorithm из статьи Глика)."""
    a = math.log(r.sigma**2)
    phi2 = r.phi**2

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (delta**2 - phi2 - v - ex)
        den = 2.0 * (phi2 + v + ex) ** 2
        return num / den - (x - a) / tau**2

    big_a = a
    if delta**2 > phi2 + v:
        big_b = math.log(delta**2 - phi2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        big_b = a - k * tau

    fa, fb = f(big_a), f(big_b)
    while abs(big_b - big_a) > CONVERGENCE_EPS:
        big_c = big_a + (big_a - big_b) * fa / (fb - fa)
        fc = f(big_c)
        if fc * fb <= 0:
            big_a, fa = big_b, fb
        else:
            fa /= 2.0
        big_b, fb = big_c, fc
    return math.exp(big_a / 2.0)


def update(
    r: Rating,
    opponents: list[Rating],
    scores: list[float],
    tau: float = DEFAULT_TAU,
    inactive_periods: float = 0.0,
) -> Rating:
    """
    Обновление рейтинга по результатам рейтингового периода.
    opponents/scores - соперники и результаты (1 победа, 0 поражение, 0.5 ничья).
    Возвращает НОВЫЙ объект Rating (старый не мутируется).
    """
    phi = r.pre_rating_rd(inactive_periods)
    if not opponents:
        # Нет игр в периоде - растёт только неопределённость
        return Rating(r.mu, phi, r.sigma, r.n_games, r.last_ts)

    work = Rating(r.mu, phi, r.sigma)

    v_inv = 0.0
    delta_sum = 0.0
    for opp, s in zip(opponents, scores):
        g_phi = _g(opp.phi)
        e = 1.0 / (1.0 + math.exp(-g_phi * (work.mu - opp.mu)))
        v_inv += g_phi**2 * e * (1.0 - e)
        delta_sum += g_phi * (s - e)

    v = 1.0 / v_inv
    delta = v * delta_sum

    sigma_new = _new_sigma(work, delta, v, tau)
    phi_star = math.sqrt(work.phi**2 + sigma_new**2)
    phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
    mu_new = work.mu + phi_new**2 * delta_sum

    return Rating(
        mu=mu_new,
        phi=phi_new,
        sigma=sigma_new,
        n_games=r.n_games + len(opponents),
        last_ts=r.last_ts,
    )


@dataclass
class PlayerPool:
    """
    Пул рейтингов игроков. Рейтинг команды = агрегат рейтингов её игроков,
    поэтому замена одного игрока (стенд-ин) корректно меняет прогноз.
    """

    tau: float = DEFAULT_TAU
    players: dict[str, Rating] = field(default_factory=dict)

    def get(self, player_id: str) -> Rating:
        if player_id not in self.players:
            self.players[player_id] = Rating()
        return self.players[player_id]

    def team_rating(self, roster: list[str]) -> Rating:
        """Композитный рейтинг состава: среднее mu, phi по правилу сумм дисперсий."""
        rs = [self.get(p) for p in roster]
        n = len(rs)
        mu = sum(x.mu for x in rs) / n
        # Дисперсия среднего независимых величин: sum(phi^2) / n^2
        phi = math.sqrt(sum(x.phi**2 for x in rs)) / n
        sigma = sum(x.sigma for x in rs) / n
        return Rating(mu=mu, phi=phi, sigma=sigma)

    def predict(self, roster_a: list[str], roster_b: list[str]) -> float:
        """P(команда A победит команду B)."""
        return expected_score(self.team_rating(roster_a), self.team_rating(roster_b))

    def record_match(
        self,
        roster_a: list[str],
        roster_b: list[str],
        a_won: bool,
        ts: float | None = None,
    ) -> None:
        """
        Обновляет рейтинги всех 10 игроков по итогу карты/матча.
        Каждый игрок 'сыграл' против композитного рейтинга состава соперника.
        """
        team_b = self.team_rating(roster_b)
        team_a = self.team_rating(roster_a)
        s_a = 1.0 if a_won else 0.0

        new_a = {
            p: update(self.get(p), [team_b], [s_a], self.tau) for p in roster_a
        }
        new_b = {
            p: update(self.get(p), [team_a], [1.0 - s_a], self.tau) for p in roster_b
        }
        for p, r in {**new_a, **new_b}.items():
            r.last_ts = ts
            self.players[p] = r
