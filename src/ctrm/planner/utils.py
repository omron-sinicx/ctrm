"""utilities of planning
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import numpy as np

from ..roadmap import TimedNode


def get_cost(
    path: list[TimedNode], goal: np.ndarray, goal_rad: float
) -> float:
    """compute path cost

    Args:
        path (list[TimedNode])
        goal (np.ndarray): goal
        goal_rad (float): goal radius

    Returns:
        float: cost

    Note:
        More precisely, this is approximation.
    """
    cost = len(path) - 1
    while (
        cost - 1 >= 0 and np.linalg.norm(path[cost - 1].pos - goal) <= goal_rad
    ):
        cost -= 1
    return cost


def get_travel_dist(path: list[TimedNode]) -> float:
    """compute travel distance

    Args:
        path (list[TimedNode]): path

    Returns:
        float: distance
    """

    return float(
        sum(
            [
                np.linalg.norm(path[i + 1].pos - path[i].pos)
                for i in range(len(path) - 1)
            ]
        )
    )
