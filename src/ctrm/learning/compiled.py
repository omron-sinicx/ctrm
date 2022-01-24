"""utilities to speedup calculations with jit
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

import numpy as np
import torch
from numba import f8, jit


@jit(f8[:, :](f8[:, :]), nopython=True)
def get_normed_vec_mag(arr_vec: np.ndarray) -> np.ndarray:
    """compute
       from [[x1, y1], [x2, y2], ...]
       to   [[normed_x1, normed_y1, mag], [normed_x2, normed_y2, mag], ...]

    Args:
        arr_vec (np.ndarray): un-normalized vector (2D)

    Returns:
         np.ndarray: normalized vector (3D)
    """
    # np.linalg.norm with axis cannot be used with numba.jit
    vec_mag = np.sqrt(np.sum(arr_vec ** 2, axis=1)).reshape(-1, 1)
    vec_mag_avoid_zero = np.where(vec_mag == 0, 1, vec_mag)
    arr_vec = arr_vec / vec_mag_avoid_zero
    return np.hstack((arr_vec, vec_mag))


@jit(
    f8[:, :, :](f8[:, :, :], f8[:, :], f8[:, :, :], f8[:], f8[:]),
    nopython=True,
)
def get_arr_others_info(
    arr_current_locs: np.ndarray,
    goals: np.ndarray,
    arr_prev_locs: np.ndarray,
    rads: np.ndarray,
    max_speeds: np.ndarray,
) -> np.ndarray:
    """get other agents relative info

    Args:
        arr_current_locs (np.ndarray): current locations
        goals (np.ndarray): goal positions for all agents
        arr_prev_locs (np.ndarray): previous locations
        rads (np.ndarray): radius for all agents
        max_speeds (np.ndarray): max speed for all agents

    Returns:
        np.ndarray:
            time *
            (self-agent -> other-agent) *
            (current_vec, goal_vec, prev_vec, rad, speed)

    Todo:
        improving readability
    """

    num_agents = goals.shape[0]
    T = arr_current_locs.shape[0]

    # get relative pos
    arr_relative_pos = np.zeros((T * num_agents * num_agents * 3, 2))
    for t in range(T):
        for i in range(num_agents):
            idx = t * (num_agents * num_agents * 3) + i * (num_agents * 3)
            current_pos = arr_current_locs[t][i]
            arr_relative_pos[idx + 0 * num_agents : idx + 1 * num_agents] = (
                arr_current_locs[t] - current_pos
            )
            arr_relative_pos[idx + 1 * num_agents : idx + 2 * num_agents] = (
                goals - current_pos
            )
            arr_relative_pos[idx + 2 * num_agents : idx + 3 * num_agents] = (
                arr_prev_locs[t] - current_pos
            )
    arr_relative_pos = get_normed_vec_mag(arr_relative_pos)

    arr_others_info = np.empty((T, num_agents * num_agents, 11))
    for t in range(T):
        for i in range(num_agents):
            idx = t * (num_agents * num_agents * 3) + i * (num_agents * 3)
            for j in range(num_agents):
                k = i * num_agents + j
                arr_others_info[t, k, 0:3] = arr_relative_pos[
                    j + 0 * num_agents + idx
                ]
                arr_others_info[t, k, 3:6] = arr_relative_pos[
                    j + 1 * num_agents + idx
                ]
                arr_others_info[t, k, 6:9] = arr_relative_pos[
                    j + 2 * num_agents + idx
                ]
                arr_others_info[t, k, 9] = rads[j]
                arr_others_info[t, k, 10] = max_speeds[j]
    return arr_others_info


@torch.jit.script
def get_self_info(
    num_agents: int, arr_others_info: torch.Tensor
) -> torch.Tensor:
    """get self-info from array of other_agents_info

    Args:
        num_agents (int): number of agents
        arr_others_info (torch.Tensor): other agents' info

    Returns:
        torch.Tensor: time * agent * (goal_vec, prev_vec, rad, max_speed)
    """
    T = arr_others_info.shape[0]
    self_idx = torch.tensor(
        [
            [
                t * num_agents * num_agents + num_agents * i + i
                for i in range(num_agents)
            ]
            for t in range(T)
        ]
    ).reshape(-1)
    return arr_others_info.reshape(-1, arr_others_info.shape[-1])[:, 3:][
        self_idx
    ].reshape(T, num_agents, -1)
