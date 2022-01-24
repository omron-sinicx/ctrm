"""utilities of roadmap construction
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from tqdm import tqdm

from ..environment import Instance, StaticObjects
from .timed_roadmap import TimedRoadmap


def valid_move(
    pos1: np.ndarray,
    pos2: np.ndarray,
    max_speed: float,
    rad: float,
    objs: StaticObjects,
) -> bool:
    """check whether the move is valid for the given agent

    Args:
        pos1 (np.ndarray): 'from' position
        pos2 (np.ndarray): 'to' position
        max_speed (float): max speed
        rad (float): radius
        objs (StaticObjects): static obstacles

    Returns:
        bool: false -> invalid
    """
    p = np.abs(pos1 - pos2)
    if any(p > max_speed):  # fast check, use Manhattan distance
        return False
    if sum(p ** 2) > max_speed ** 2:  # exact check
        return False
    return not objs.collide_continuous_sphere(pos1, pos2, rad)


def get_fixed_structured_roadmaps(
    ins: Instance, T: int, arr_locs: list[list[np.ndarray]], verbose: int = 0,
) -> list[TimedRoadmap]:
    """get timed roadmaps for fixed structure beyond one timestep

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        arr_locs (list[list[np.ndarray]]): sampled locations
        verbose (:obj:`int`, optional): whether to print additional info

    Returns:
        list[TimedRoadmap]: timed roadmaps until timestep T
    """
    # initialize
    trms: list[TimedRoadmap] = [
        TimedRoadmap(ins.starts[i]) for i in range(ins.num_agents)
    ]

    for i in tqdm(
        range(ins.num_agents),
        desc="generate timed roadmap",
        disable=(0 == verbose),
    ):
        locs = arr_locs[i]

        def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
            # define valid moves
            return valid_move(
                pos1, pos2, ins.max_speeds[i], ins.rads[i], ins.objs
            )

        # append to body
        trms[i].append_fixed_strucutre(locs, T, valid_edge)

    return trms


def get_common_roadmaps(
    ins: Instance,
    T: int,
    locs_common: list[np.ndarray],
    adjs_common: Optional[list[list[int]]] = None,
) -> list[TimedRoadmap]:
    """get common timed roadmaps for all agents

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        locs_common (list[np.ndarray]): sampled locations
        adjs_common (:obj:`Optional[list[list[int]]]`, optional):
            adjacency info

    Returns:
        list[TimedRoadmap]: timed roadmaps until timestep T
    """

    assert all([rad == ins.rads[0] for rad in ins.rads])
    assert all([speed == ins.max_speeds[0] for speed in ins.max_speeds])

    trms: list[TimedRoadmap] = [
        TimedRoadmap(ins.starts[i]) for i in range(ins.num_agents)
    ]

    def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
        return valid_move(pos1, pos2, ins.max_speeds[0], ins.rads[0], ins.objs)

    # compute common adjacency
    if adjs_common is None:
        adjs_common = [[i] for i in range(len(locs_common))]
        for i in range(len(locs_common)):
            for j in range(i + 1, len(locs_common)):
                if valid_edge(locs_common[i], locs_common[j]):
                    adjs_common[i].append(j)
                    adjs_common[j].append(i)

    for i in range(ins.num_agents):
        # add goal
        locs = locs_common + [ins.goals[i]]

        # compute custom adjacency
        adjs = copy.deepcopy(adjs_common) + [[]]
        for j, loc in enumerate(locs_common):
            if valid_edge(loc, ins.goals[i]):
                adjs[j].append(len(locs_common))
                adjs[-1].append(j)

        trms[i].append_fixed_strucutre(locs, T, valid_edge, adjs)

    return trms
