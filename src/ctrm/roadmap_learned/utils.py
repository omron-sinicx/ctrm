"""utilities for generation of CTRMs
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import numpy as np
from numba import f8, jit

from ..environment import Instance
from ..roadmap import TimedNode, TimedRoadmap
from ..roadmap.utils import valid_move


@jit(f8[:](f8[:, :], f8[:]), nopython=True)
def get_dist_arr(cands_pos: np.ndarray, loc: np.ndarray) -> np.ndarray:
    return np.sum((cands_pos - loc) ** 2, axis=1)


def merge_samples(
    loc: np.ndarray,
    t: int,
    agent: int,
    trm: TimedRoadmap,
    ins: Instance,
    merge_distance: float = 0.01,
) -> np.ndarray:
    """find compatible sample, otherwise return loc

    Args:
        loc (np.ndarray): location
        t (int): timestep
        agent (int): target agent
        trm (TimedRoadmap): target timed roadmap
        ins (Instance): instance
        merge_distance (:obj:`float`, optional):
            distance regarding as spatially close enough

    Returns:
        np.ndarray: location of compatible sample if found, otherwise loc

    Todo:
        use efficient set operation
    """
    rad = ins.rads[agent]
    max_speed = ins.max_speeds[agent]
    goal = ins.goals[agent]

    # get necessary distance
    cands_pos_arr = [u.pos for u in trm.V[t - 1]]  # parents
    if t + 1 <= len(trm.V) - 1:
        cands_pos_arr += [u.pos for u in trm.V[t + 1]]  # children
    if len(trm.V) > t:
        cands_pos_arr += [u.pos for u in trm.V[t]]  # merge
    dist_arr = get_dist_arr(np.array(cands_pos_arr), loc)

    # compute parents
    offset = len(trm.V[t - 1])
    parents_cands_index = np.where(dist_arr[:offset] <= max_speed ** 2)[0]
    parents = [
        i
        for i in parents_cands_index
        if not ins.objs.collide_continuous_sphere(
            trm.V[t - 1][i].pos, loc, rad
        )
    ]
    set_loc_parents = set(parents)

    # compute children
    if t + 1 <= len(trm.V) - 1:
        children_cands_index = np.where(
            dist_arr[offset : offset + len(trm.V[t + 1])] <= max_speed ** 2
        )[0]
        children = [
            i
            for i in children_cands_index
            if not ins.objs.collide_continuous_sphere(
                trm.V[t + 1][i].pos, loc, rad
            )
        ]
    else:
        children = []
    set_loc_children = set(children)

    if len(trm.V) > t:
        merge_cands_idx = np.where(
            dist_arr[-len(trm.V[t]) :] <= merge_distance ** 2
        )[0]

        # get heuristics
        h_loc = sum((loc - goal) ** 2)

        for u_ind in merge_cands_idx:
            u = trm.V[t][u_ind]
            u_parents = trm.get_parents(u)
            u_children = trm.E[t][u.index]
            set_u_parents = set(u_parents)
            set_u_children = set(u_children)

            if (
                set_u_parents == set_loc_parents
                and set_u_children == set_loc_children
            ):
                # merge to better one
                h_u = sum((u.pos - goal) ** 2)
                if h_loc < h_u:
                    # replace u by loc
                    trm.V[t][u.index] = TimedNode(t, u.index, loc)
                    return loc
                else:
                    # abandon loc
                    return u.pos

            if (
                set_u_parents >= set_loc_parents
                and set_u_children >= set_loc_children
            ):
                # abandon loc
                return u.pos

            if (
                set_u_parents <= set_loc_parents
                and set_u_children <= set_loc_children
            ):
                # replace u by loc
                trm.V[t][u.index] = TimedNode(t, u.index, loc)
                # append additional edge, children
                trm.E[t][u.index] += list(set_loc_children - set_u_children)
                # append parents
                for p in set_loc_parents - set_u_parents:
                    trm.E[t - 1][p].append(u.index)
                return loc

    # append new sample
    trm.append_sample(loc=loc, t=t, parents=parents, children=children)
    return loc


def format_trms(ins: Instance, trms: list[TimedRoadmap]) -> None:
    """align length of timed roadmaps

    Args:
        ins (Instance): instance
        trms (list[TimedRoadmap]): timed roadmaps
    """
    T = max([len(trm.V) for trm in trms]) - 1
    for i, trm in enumerate(trms):

        def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
            return valid_move(
                pos1, pos2, ins.max_speeds[i], ins.rads[i], ins.objs
            )

        # technical point, add one additional layer
        trm.extend_until(T + 1, valid_edge)


def append_goals(ins: Instance, trms: list[TimedRoadmap]) -> None:
    """append goals to timed roadmaps

    Args:
        ins (Instance): instance
        trms (list[TimedRoadmap]): timed roadmaps
    """
    for i, (trm, goal) in enumerate(zip(trms, ins.goals)):

        def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
            return valid_move(
                pos1, pos2, ins.max_speeds[i], ins.rads[i], ins.objs
            )

        for t in range(1, len(trm.V)):
            trm.append_sample(goal, t, valid_edge)
