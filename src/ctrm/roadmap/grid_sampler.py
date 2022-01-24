"""grid sampling
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ..environment import Instance
from .timed_roadmap import TimedRoadmap
from .utils import (get_common_roadmaps, get_fixed_structured_roadmaps,
                    valid_move)


def get_grid(size: int, rad: float, ins: Instance) -> list[np.ndarray]:
    """grid sampling

    Args:
        size (int): size x size grid will be constructed
        rad (float): agent radius
        ins (Instance): instance

    Returns:
        list[np.ndarray]: locations
    """

    return [
        np.array(s)
        for s in product(np.linspace(rad, 1 - rad, size), repeat=ins.dim)
        if not ins.objs.collide_sphere(np.array(s), rad)
    ]


def get_timed_roadmaps_grid(
    ins: Instance, T: int, size: int, verbose=0,
) -> list[TimedRoadmap]:
    """grid respective grid ramps

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        size (int): size x size grid will be constructed
        verbose (:obj:`int`, optional): >0 -> print additional info

    Returns:
        list[np.ndarray]: locations
    """
    arr_locs = [
        get_grid(size, ins.rads[i], ins) + [ins.goals[i]]
        for i in range(ins.num_agents)
    ]
    return get_fixed_structured_roadmaps(ins, T, arr_locs, verbose)


def get_timed_roadmaps_grid_common(
    ins: Instance, T: int, size: int,
) -> list[TimedRoadmap]:
    """[deprecated] get grid roadmap shared by all agents

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        size (int): size x size grid will be constructed

    Returns:
        list[np.ndarray]: locations

    Note:
        use get_timed_roadmaps_grid_common_2d_fast in 2d environment
    """
    if ins.dim == 2:
        return get_timed_roadmaps_grid_common_2d_fast(ins, T, size)
    return get_common_roadmaps(ins, T, get_grid(size, ins.rads[0], ins))


def get_timed_roadmaps_grid_common_2d_fast(
    ins: Instance, T: int, size: int,
) -> list[TimedRoadmap]:
    """get grid roadmap shared by all agents

    Args:
        ins (Instance): instance
        T (int): assumed makespan
        size (int): size x size grid will be constructed

    Returns:
        list[np.ndarray]: locations

    Note:
        remove redundant connectivity check of "get_timed_roadmaps_grid_common"
    """
    locs_common: list[np.ndarray] = []

    rad = ins.rads[0]
    speed = ins.max_speeds[0]
    cell_len = 1 / size
    reachable_len = int(speed / cell_len)

    def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
        return valid_move(pos1, pos2, speed, rad, ins.objs)

    # create vertexes
    indexes = np.full((size, size), -1, dtype=np.int32)
    sample_cnt = 0
    for i, s in enumerate(product(np.linspace(rad, 1 - rad, size), repeat=2)):
        if ins.objs.collide_sphere(np.array(s), rad):
            continue
        indexes[i // size, i % size] = sample_cnt
        locs_common.append(np.array(s))
        sample_cnt += 1

    # create adjacency
    adjs_common = [[i] for i in range(len(locs_common))]
    for i, j in product(range(size), repeat=2):
        idx = indexes[i, j]
        if idx == -1:  # not sample
            continue
        loc = locs_common[idx]
        # create kernel
        for _i in range(
            # max(0, i - reachable_len), min(i + reachable_len, size - 1) + 1
            i,
            min(i + reachable_len, size - 1) + 1,  # the above is redundant
        ):
            for _j in range(
                max(0, j - reachable_len), min(j + reachable_len, size - 1) + 1
            ):
                if _i == i and _j == j:  # skip itself
                    continue
                _idx = indexes[_i, _j]
                if _idx < idx:  # not sample or avoid duplication
                    continue
                if valid_edge(loc, locs_common[_idx]):
                    adjs_common[idx].append(_idx)
                    adjs_common[_idx].append(idx)

    return get_common_roadmaps(ins, T, locs_common, adjs_common)
