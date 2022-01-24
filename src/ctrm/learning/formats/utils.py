"""utilities for formatting
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import numpy as np


def get_relative_pos(
    target_pos: np.ndarray, current_pos: np.ndarray
) -> np.ndarray:
    """get relative position

    Args:
        target_pos (np.ndarray): target position
        current_pos (np.ndarray): base position

    Returns:
        np.ndarray: normalized_x, normalized_y, magnitude
    """
    vec = target_pos - current_pos
    vec_mag = np.linalg.norm(vec)
    # avoid dividing zero
    if vec_mag > 0:
        vec = vec / vec_mag
    return np.hstack([vec, vec_mag])


def get_goalvec_mag(**kwargs) -> np.ndarray:
    """get goal vector"""
    return get_relative_pos(kwargs["goal"], kwargs["current_pos"])


def get_nextvec_mag(**kwargs) -> np.ndarray:
    """get next vector"""
    return get_relative_pos(kwargs["next_pos"], kwargs["current_pos"])
