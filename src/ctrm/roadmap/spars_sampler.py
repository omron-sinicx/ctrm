"""SPARS sampling
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX

Ref:
- Dobson, A., Krontiris, A., & Bekris, K. E. (2013).
  Sparse roadmap spanners.
  In Algorithmic Foundations of Robotics X (pp. 279-296).
  Springer, Berlin, Heidelberg.

- Sucan, I. A., Moll, M., & Kavraki, L. E. (2012).
  The open motion planning library.
  IEEE Robotics & Automation Magazine, 19(4), 72-82.

- https://ompl.kavrakilab.org/classompl_1_1geometric_1_1SPARS.html
"""

from __future__ import annotations

import numpy as np
from spars_wrapper import getSparsRoadmap2d
from tqdm import tqdm

from ..environment import Instance
from .timed_roadmap import TimedRoadmap
from .utils import get_common_roadmaps, valid_move


def get_spars_samples_w_adj(
    ins: Instance,
    rad: float,
    speed: float,
    obs_dict: list[dict],
    sparse_delta_fraction: float = 0.1,
    dense_delta_fraction: float = 0.01,
    stretch_factor: float = 1.3,
    time_limit_sec: float = 10,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """call SPARS wrapper written in c++

    Args:
        ins (Instance): instance
        rad (float): radius
        speed (float): max speed
        obs_dict (list[dict]): static objects
        sparse_delta_fraction (:obj:`float`, optional): params in SPARS
        dense_delta_fraction (:obj:`float`, optional): params in SPARS
        stretch_factor (:obj:`float`, optional): params in SPARS
        time_limit_sec (:obj:`float`, optional): params in SPARS

    Returns:
        list[np.ndarray]: locations
        list[list[int]]]: adjacency
    """
    (
        spars,
        cnt_continuous_collide,
        elapsed_continuous_collide,
        cnt_static_collide,
        elapsed_static_collide,
    ) = getSparsRoadmap2d(
        rad,
        speed,
        sparse_delta_fraction,
        dense_delta_fraction,
        stretch_factor,
        10000000,
        time_limit_sec,
        obs_dict,
        rad,  # lower bound
        1 - rad,  # upper bound
    )
    locs, adjs = [], []
    for s in spars:
        locs.append(np.array(s[:2]))
        adjs.append(s[-1])
    for i in range(len(adjs)):
        adjs[i].append(i)

    # update collision counts
    ins.objs.cnt_continuous_collide += cnt_continuous_collide
    ins.objs.time_continuous_collide += elapsed_continuous_collide
    ins.objs.cnt_static_collide += cnt_static_collide
    ins.objs.time_static_collide += elapsed_continuous_collide

    return locs, adjs


def get_timed_roadamaps_SPARS_2d(
    ins: Instance,
    T: int,
    sparse_delta_fraction: float = 0.1,
    dense_delta_fraction: float = 0.1,
    stretch_factor: float = 1.3,
    time_limit_sec: float = 1.0,
    verbose: int = 0,
) -> list[TimedRoadmap]:
    """get SPARS roadmaps, respective for each agent

    Args
        ins (Instance): instance
        T (int): assumed makespan
        sparse_delta_fraction (:obj:`float`, optional): params of SPARS
        dense_delta_fraction (:obj:`float`, optional): params of SPAR
        stretch_factor (:obj:`float`, optional): params of SPAR
        time_limit_sec (:obj:`float`, optional): params of SPAR
        verbose (:obj:`int`, optional): >0 -> print additional info

    Returns:
        list[TimedRoadmap]: timed roadmaps
    """

    assert ins.dim == 2
    trms: list[TimedRoadmap] = [
        TimedRoadmap(ins.starts[i]) for i in range(ins.num_agents)
    ]
    obs_dict = ins.objs.get_dict_for_spars()

    for i in tqdm(range(ins.num_agents)):
        # get spars map and adjacency
        locs, adjs = get_spars_samples_w_adj(
            ins,
            ins.rads[i],
            ins.max_speeds[i],
            obs_dict,
            sparse_delta_fraction,
            dense_delta_fraction,
            stretch_factor,
            time_limit_sec,
        )

        # define valid moves
        def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
            return valid_move(
                pos1, pos2, ins.max_speeds[i], ins.rads[i], ins.objs
            )

        # append goal
        locs.append(ins.goals[i])
        adjs.append([])
        for j, loc in enumerate(locs[:-1]):
            if valid_edge(loc, ins.goals[i]):
                adjs[j].append(len(locs) - 1)
                adjs[-1].append(j)

        trms[i].append_fixed_strucutre(locs, T, valid_edge, adjs)

    return trms


def get_timed_roadamaps_SPARS_2d_common(
    ins: Instance,
    T: int,
    sparse_delta_fraction: float = 0.1,
    dense_delta_fraction: float = 0.01,
    stretch_factor: float = 1.3,
    time_limit_sec: float = 10.0,
) -> list[TimedRoadmap]:
    """get SPARS roadmaps shared by all agents

    Args
        ins (Instance): instance
        T (int): assumed makespan
        sparse_delta_fraction (:obj:`float`, optional): params of SPARS
        dense_delta_fraction (:obj:`float`, optional): params of SPAR
        stretch_factor (:obj:`float`, optional): params of SPAR
        time_limit_sec (:obj:`float`, optional): params of SPAR

    Returns:
        list[TimedRoadmap]: timed roadmaps
    """

    assert ins.dim == 2
    locs_common, adjs_common = get_spars_samples_w_adj(
        ins,
        ins.rads[0],
        ins.max_speeds[0],
        ins.objs.get_dict_for_spars(),
        sparse_delta_fraction,
        dense_delta_fraction,
        stretch_factor,
        time_limit_sec,
    )
    return get_common_roadmaps(ins, T, locs_common, adjs_common)
