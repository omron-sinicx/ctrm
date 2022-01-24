"""obstacle
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import fcl
import numpy as np
from skimage.draw import disk

from .fcl_utils import get_box, get_sphere


class Obstacle(metaclass=ABCMeta):
    """template of static obstacles"""

    pos: np.ndarray
    circumradius: float  # used in fast collision checking

    @abstractmethod
    def get(self) -> fcl.CollisionObject:
        """get FCL object"""
        pass

    @abstractmethod
    def get_dict_for_spars(self) -> dict:
        """used for cpp_wrapper"""
        pass

    @abstractmethod
    def draw_2d(self, map_size: int) -> np.ndarray:
        """return numpy grid array, used for creating FOV"""
        pass


@dataclass(frozen=True)
class ObstacleSphere(Obstacle):
    """static obstacle (sphere)"""

    pos: np.ndarray  # center position
    rad: float  # radius

    def __post_init__(self):
        object.__setattr__(self, "circumradius", self.rad)

    def get(self) -> fcl.CollisionObject:
        return get_sphere(self.pos, self.rad)

    def get_dict_for_spars(self) -> dict:
        """for c++ wrapper (SPARS)"""
        return {
            "type": "sphere",
            "rad": self.rad,
            "x": self.pos[0],
            "y": self.pos[1],
            "z": self.pos[2] if self.pos.shape[0] == 3 else 0,
        }

    def draw_2d(self, map_size: int) -> np.ndarray:
        """draw 2d image, used in FOV creating

        Args:
            map_size (int): map size

        Returns:
            np.ndarray: 2D array
        """

        shape = (map_size, map_size)
        img = np.zeros(shape, dtype=np.float32)
        X = int(map_size * self.pos[0])
        Y = int(map_size * self.pos[1])
        R = int(map_size * self.rad)
        rr, cc = disk((X, Y), R, shape=shape)
        img[rr, cc] = 1.0

        return img


@dataclass(frozen=True)
class ObstacleBox(Obstacle):
    """deprecated: static obstacle (square)"""

    pos: np.ndarray  # center position
    size: np.ndarray  # x-size, y-size, (z-size)

    def __post_init__(self):
        object.__setattr__(
            self, "circumradius", np.sqrt(sum((self.size / 2) ** 2))
        )

    def get(self) -> fcl.CollisionObject:
        """get FCL object"""
        return get_box(self.pos, self.size)

    def get_dict_for_spars(self):
        """for c++ wrapper (SPARS)"""
        return {
            "type": "box",
            "size_x": self.size[0],
            "size_y": self.size[1],
            "size_z": self.size[2] if self.size.shape[0] == 3 else 0,
            "x": self.pos[0],
            "y": self.pos[1],
            "z": self.pos[2] if self.pos.shape[0] == 3 else 0,
        }

    def draw_2d(self, map_size: int) -> np.ndarray:
        """
        Todo:
            This function is not coded yet.
        """
        pass
