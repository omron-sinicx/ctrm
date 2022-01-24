"""random and square sampling
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX

Note: "random sampling" is equivalent to a simplified version of PRM.
Ref:
- Karaman, S., & Frazzoli, E. (2011).
  Sampling-based algorithms for optimal motion planning.
  The international journal of robotics research (IJRR)
"""

from __future__ import annotations

import numpy as np

from ..environment import Instance
from .timed_roadmap import TimedRoadmap
from .utils import (get_common_roadmaps, get_fixed_structured_roadmaps,
                    valid_move)


def get_random_samples(
    num: int, rad: float, ins: Instance, wo_invalid_samples: bool = True,
) -> list[np.ndarray]:
    """random sampling

    Args:
        num (int): number of samples
        rad (float): agent radius
        ins (Instance): instance
        wo_invalid_samples (:obj:`bool`, optional):
            whether to include invalid samples in counts

    Returns:
        list[np.ndarray]: locations
    """
    locs: list[np.ndarray] = []

    if wo_invalid_samples:
        while len(locs) < num:
            pos = np.random.random(ins.dim) * (1 - 2 * rad) + rad
            if not ins.objs.collide_sphere(pos, rad):
                locs.append(pos)
    else:
        for _ in range(num):
            pos = np.random.random(ins.dim) * (1 - 2 * rad) + rad
            if not ins.objs.collide_sphere(pos, rad):
                locs.append(pos)

    return locs


def get_timed_roadmaps_random(
    ins: Instance,
    T: int,
    num: int,
    wo_invalid_samples: bool = True,
    verbose: int = 0,
) -> list[TimedRoadmap]:
    """get respective random roadmaps

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        num (int): number of samples
        wo_invalid_samples (:obj:`bool`, optional):
            whether to include invalid samples in counts
        verbose (:obj:`int`, optional): >0 -> print additional info

    Returns:
        list[TimedRoadmap]: timed roadmaps
    """
    arr_locs = [
        get_random_samples(num, ins.rads[i], ins, wo_invalid_samples)
        + [ins.goals[i]]
        for i in range(ins.num_agents)
    ]
    return get_fixed_structured_roadmaps(ins, T, arr_locs, verbose)


def get_timed_roadmaps_fully_random(
    ins: Instance, T: int, num: int, wo_invalid_samples: bool = True,
) -> list[TimedRoadmap]:
    """get respective random roadmaps for each timestep

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        num (int): number of samples
        wo_invalid_samples (:obj:`bool`, optional):
            whether to include invalid samples in counts

    Returns:
        list[TimedRoadmap]: timed roadmaps
    """
    trms: list[TimedRoadmap] = [
        TimedRoadmap(ins.starts[i]) for i in range(ins.num_agents)
    ]

    for t in range(1, T + 1):
        for i in range(ins.num_agents):
            # create samples with goal
            locs = get_random_samples(
                num, ins.rads[i], ins, wo_invalid_samples
            ) + [ins.goals[i]]

            # define valid moves
            def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
                return valid_move(
                    pos1, pos2, ins.max_speeds[i], ins.rads[i], ins.objs
                )

            # append to timed roadmap
            trms[i].append_samples(locs, t, valid_edge)

    return trms


def get_timed_roadmaps_random_common(
    ins: Instance, T: int, num: int, wo_invalid_samples: bool = True,
) -> list[TimedRoadmap]:
    """get random roadmap shared by all agents

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        num (int): number of samples
        wo_invalid_samples (:obj:`bool`, optional):
            whether to include invalid samples in counts

    Returns:
        list[TimedRoadmap]: timed roadmaps
    """
    return get_common_roadmaps(
        ins, T, get_random_samples(num, ins.rads[0], ins, wo_invalid_samples)
    )


def get_random_samples_rect(
    sample_rate_vs_diag_speed: float,
    ins: Instance,
    agent: int,
    margin_rate_vs_speed: float = 0,
    wo_invalid_samples: bool = True,
) -> list[np.ndarray]:
    """square sampling

    Args:
        sample_rate_vs_diag_speed: (float): density parameter
        ins (Instance): instance
        agent (int): target agent id
        margin_rate_vs_speed (:obj:`float`, optional):
            margin of the diagonal line
        wo_invalid_samples (:obj:`bool`, optional):
            whether to include invalid samples in counts

    Returns:
        list[np.ndarray]: locations
    """
    start = ins.starts[agent]
    goal = ins.goals[agent]
    speed = ins.max_speeds[agent]
    rad = ins.rads[agent]
    midpoint = (start + goal) / 2
    diag = np.linalg.norm(goal - start)
    num = int((diag / speed) * sample_rate_vs_diag_speed)
    size = (diag + speed * margin_rate_vs_speed) / np.sqrt(2)
    vec = goal - start
    theta = np.arctan2(vec[1], vec[0]) + np.pi / 4
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    def get_new_sample(_num):
        points = (
            np.dot(R, np.random.rand(ins.dim, _num) - 0.5).T * size + midpoint
        )
        return np.clip(points, 0, 1)

    locs = []

    if wo_invalid_samples:
        cands = get_new_sample(num)
        for pos in cands:
            if not ins.objs.collide_sphere(pos, rad):
                locs.append(pos)
    else:
        while len(locs) < num:
            pos = get_new_sample(1)[0]
            if not ins.objs.collide_sphere(pos, rad):
                locs.append(pos)
    return locs


def get_timed_roadmaps_random_rect(
    ins: Instance,
    T: int,
    sample_rate_vs_diag_speed: float,
    margin_rate_vs_speed: float = 0.05,
    wo_invalid_samples: bool = True,
    verbose: int = 0,
) -> list[TimedRoadmap]:
    """construct roadmaps by square sampling

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        sample_rate_vs_diag_speed: (float): density parameter
        margin_rate_vs_speed (:obj:`float`, optional):
            margin of the diagonal line
        wo_invalid_samples (:obj:`bool`, optional):
            whether to include invalid samples in counts
        verbose (:obj:`int`, optional): >0 -> print additional info

    Returns:
        list[TimedRoadmap]: timed roadmaps
    """
    arr_locs = [
        get_random_samples_rect(
            sample_rate_vs_diag_speed=sample_rate_vs_diag_speed,
            ins=ins,
            agent=i,
            margin_rate_vs_speed=margin_rate_vs_speed,
            wo_invalid_samples=wo_invalid_samples,
        )
        + [ins.goals[i]]
        for i in range(ins.num_agents)
    ]
    return get_fixed_structured_roadmaps(ins, T, arr_locs, verbose)
