"""Lévy-flight enhanced Sparrow Search Algorithm (LF-SSA).

The public entry point remains ``SSA_Simple_Improved`` for compatibility with
the existing VMD optimisation scripts.  The implementation adds elitist greedy
selection, adaptive Lévy exploration, diversity recovery and robust bounds
handling.  These changes address the original implementation's population
overwrite behaviour, which could discard a better candidate at every round.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def levy_flight(dim: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Mantegna Lévy step with bounded tails for numerical stability."""
    sigma_u = (
        math.gamma(1.0 + beta)
        * math.sin(math.pi * beta / 2.0)
        / (
            math.gamma((1.0 + beta) / 2.0)
            * beta
            * 2.0 ** ((beta - 1.0) / 2.0)
        )
    ) ** (1.0 / beta)
    u = rng.normal(size=dim) * sigma_u
    v = rng.normal(size=dim)
    step = u / (np.abs(v) ** (1.0 / beta) + 1e-12)
    return step / (1.0 + np.abs(step))


def _as_bounds(
    lb: float | Sequence[float], ub: float | Sequence[float], dim: int
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    if np.any(upper <= lower):
        raise ValueError("Each upper bound must be greater than its lower bound.")
    return lower, upper


def SSA_Simple_Improved(
    pop: int,
    Max_iter: int,
    lb: float | Sequence[float],
    ub: float | Sequence[float],
    dim: int,
    fobj: Callable[[np.ndarray], float],
    seed: int | None = None,
):
    """Optimise a minimisation objective with LF-SSA.

    Parameters are backward compatible with the prior function. ``seed`` is
    optional and enables reproducible experiments without modifying global
    NumPy state.
    """
    if pop < 4:
        raise ValueError("pop must be at least 4.")
    if Max_iter < 1:
        raise ValueError("Max_iter must be positive.")

    rng = np.random.default_rng(seed)
    lower, upper = _as_bounds(lb, ub, dim)
    span = upper - lower

    def evaluate(position: np.ndarray) -> float:
        try:
            value = float(fobj(position))
        except Exception:
            return float("inf")
        return value if np.isfinite(value) else float("inf")

    positions = rng.uniform(lower, upper, size=(pop, dim))
    fitness = np.asarray([evaluate(x) for x in positions])
    order = np.argsort(fitness)
    positions, fitness = positions[order], fitness[order]

    best_position = positions[0].copy()
    best_score = float(fitness[0])
    curve = np.empty(Max_iter, dtype=float)
    history = np.empty((Max_iter, dim), dtype=float)
    no_improvement = 0

    for iteration in range(Max_iter):
        progress = (iteration + 1.0) / Max_iter
        producer_count = max(1, int(round(pop * (0.20 + 0.20 * progress))))
        scout_count = max(1, int(round(pop * (0.12 + 0.08 * (1.0 - progress)))))
        safety_threshold = 0.80 - 0.35 * progress
        levy_scale = 0.10 * (1.0 - progress) + 0.01
        local_scale = 0.08 * (1.0 - progress) + 0.005

        proposals = positions.copy()
        alarm = rng.random()

        # Producers alternate between exploitation and Lévy exploration.
        for index in range(producer_count):
            current = positions[index]
            if alarm < safety_threshold:
                attraction = rng.uniform(0.25, 0.75)
                proposals[index] = (
                    current + attraction * (best_position - current)
                    + rng.normal(size=dim) * local_scale * span
                )
            else:
                proposals[index] = (
                    current
                    + levy_flight(dim, beta=1.5, rng=rng)
                    * levy_scale
                    * (np.abs(best_position - current) + 0.15 * span)
                )

        # Followers use the best solution while the weaker half retains broad
        # exploration around the worst region.
        midpoint = producer_count + (pop - producer_count) // 2
        for index in range(producer_count, pop):
            current = positions[index]
            if index < midpoint:
                proposals[index] = (
                    best_position
                    + rng.normal(size=dim) * (0.05 + 0.12 * (1.0 - progress))
                    * (np.abs(current - best_position) + 0.05 * span)
                )
            else:
                proposals[index] = (
                    current
                    + rng.normal(size=dim) * (positions[-1] - current)
                    / (index + 1.0)
                    + levy_flight(dim, beta=1.5, rng=rng)
                    * 0.5
                    * levy_scale
                    * span
                )

        # Scouts react to stagnation and preserve diversity around the elite.
        scout_indices = rng.choice(pop, size=scout_count, replace=False)
        for index in scout_indices:
            if fitness[index] > fitness[0]:
                proposals[index] = (
                    best_position
                    + rng.normal(size=dim)
                    * (np.abs(positions[index] - best_position) + 0.03 * span)
                )
            else:
                proposals[index] = positions[index] + levy_flight(
                    dim, beta=1.5, rng=rng
                ) * levy_scale * span

        proposals = np.clip(proposals, lower, upper)
        proposal_fitness = np.asarray([evaluate(x) for x in proposals])

        # Greedy selection is essential: never replace an individual with a
        # worse proposal, then explicitly retain the historical global elite.
        accepted = proposal_fitness <= fitness
        positions[accepted] = proposals[accepted]
        fitness[accepted] = proposal_fitness[accepted]

        current_index = int(np.argmin(fitness))
        current_score = float(fitness[current_index])
        if current_score < best_score:
            best_score = current_score
            best_position = positions[current_index].copy()
            no_improvement = 0
        else:
            no_improvement += 1

        # A small restart of the weakest sparrows prevents late convergence to
        # an alpha/K plateau while retaining the current elite.
        if no_improvement >= max(3, Max_iter // 5):
            restart_count = max(1, pop // 5)
            restart_indices = np.argsort(fitness)[-restart_count:]
            positions[restart_indices] = rng.uniform(
                lower, upper, size=(restart_count, dim)
            )
            fitness[restart_indices] = np.asarray(
                [evaluate(x) for x in positions[restart_indices]]
            )
            no_improvement = 0

        worst_index = int(np.argmax(fitness))
        if best_score < fitness[worst_index]:
            positions[worst_index] = best_position.copy()
            fitness[worst_index] = best_score

        order = np.argsort(fitness)
        positions, fitness = positions[order], fitness[order]
        curve[iteration] = best_score
        history[iteration] = best_position

    return best_score, best_position, curve, history
