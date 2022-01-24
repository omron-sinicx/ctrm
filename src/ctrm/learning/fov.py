"""helper functions for field of view (FOV)
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import math

import cost_to_go_wrapper
import numpy as np
from skimage.draw import disk


def get_map_coord_2d(pos: np.ndarray, map_size: int) -> tuple[int, int]:
    """compute x, y coordination in grid

    Args:
        pos (np.ndarray): position in [0, 1]^2
        map_size (int): map size, e.g., 64

    Returns:
        int: x
        int: y
    """
    x = int(pos[0] * map_size)
    y = int(pos[1] * map_size)
    return (
        x if x != map_size else map_size - 1,
        y if y != map_size else map_size - 1,
    )


def get_approx_cost_to_go_matrix_2d(
    goal: np.ndarray, occupancy_map: np.ndarray, rad: float
) -> np.ndarray:
    """obtain cost-to-go matrix

    Args:
        goal (np.ndarray): goal position
        occupancy_map (np.ndarray): 2D occupancy map
        rad (float): agent radius

    Returns:
        np.ndarray: cost-to-go matrix
    """
    map_size = occupancy_map.shape[0]
    rad_int = math.ceil(rad * map_size)
    occupancy_agent = np.zeros((rad_int * 2 + 1, rad_int * 2 + 1), dtype="int")
    occupancy_agent[
        disk(
            (rad_int, rad_int),
            rad_int,
            shape=(rad_int * 2 + 1, rad_int * 2 + 1),
        )
    ] = 1

    # use c++ wrapper
    cost_to_go = cost_to_go_wrapper.getCostToGo2D(
        goal, occupancy_map, occupancy_agent,
    ).astype(np.float32)
    cost_to_go[cost_to_go == 65535] = np.inf  # replace max value
    return cost_to_go


def get_fov2d_occupancy_cost(
    current_pos: np.ndarray,
    occupancy_map: np.ndarray,
    cost_to_go: np.ndarray,
    fov_size: int,
    flatten: bool = False,
) -> np.ndarray:
    """get FOV (occupancy and cost-to-go)

    Args:
        current_pos (np.ndarray): current position
        occupancy_map (np.ndarray): occupancy map
        cost_to_go (np.ndarray): cost-to-go map
        fov_size (int): fov size
        flatten (bool): true -> 1D vector, false -> 3D vector

    Returns:
        np.ndarray:
    """
    return cost_to_go_wrapper.getFov2dOccupancyCost(
        current_pos, occupancy_map, cost_to_go, fov_size, flatten
    )
