"""collection of static obstacles
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import fcl
import numpy as np

from ..utils import stop_watch
from .fcl_utils import continuous_collide_spheres, get_sphere
from .obstacle import Obstacle


class StaticObjects:
    """collection of static obstacles

    Attributes:
        obs (list[Obstacle]):
        objs (list[fcl.CollisionObject]): FCL objects
        manager (fcl.DynamicAABBTreeCollisionManager): collision check manager
        cnt_static_collide (int): count vertex collision checks
        time_static_collide (float): measure time for vertex collision check
        cnt_continuous_collide (int): count edge collision checks
        time_continuous_collide (float): measure time for edge collision check
    """

    def __init__(self, obs: list[Obstacle]):
        self.obs: list[Obstacle] = obs
        self.objs: list[fcl.CollisionObject] = [o.get() for o in obs]
        self.manager: fcl.DynamicAABBTreeCollisionManager = (
            fcl.DynamicAABBTreeCollisionManager()
        )
        self.manager.registerObjects(self.objs)
        self.manager.setup()

        # to measure the effort of collision check
        self.cnt_static_collide: int = 0
        self.time_static_collide: float = 0
        self.cnt_continuous_collide: int = 0
        self.time_continuous_collide: float = 0

    def collide_sphere(self, pos: np.ndarray, rad: float) -> bool:
        """collision check for static sphere

        Args:
            pos (np.ndarray): position
            rad (float): radius

        Returns:
            bool: true -> collide
        """
        res, elapsed = self.__collide_sphere(pos, rad)
        self.cnt_static_collide += 1
        self.time_continuous_collide += elapsed
        return res

    def collide_continuous_sphere(
        self, pos1: np.ndarray, pos2: np.ndarray, rad: float
    ) -> bool:
        """collision check for dynamic sphere

        Args:
            pos1 (np.ndarray): 'from' position
            pos2 (np.ndarray): 'to' position
            rad (float): radius

        Returns:
            bool: true -> collide
        """
        res, elapsed = self.__collide_continuous_sphere(pos1, pos2, rad)
        self.cnt_continuous_collide += 1
        self.time_continuous_collide += elapsed
        return res

    def __collide(self, obj: fcl.CollisionObject) -> bool:
        """private func, check static collision via FCL

        Args:
            obj (fcl.CollisionObject): FCL object

        Returns:
            bool: true -> collide
        """
        req = fcl.CollisionRequest()
        rdata = fcl.CollisionData(req)
        self.manager.collide(obj, rdata, fcl.defaultCollisionCallback)
        return rdata.result.is_collision

    @stop_watch(verbose=False)
    def __collide_sphere(self, pos: np.ndarray, rad: float) -> bool:
        """private func, check static collision of sphere object

        Args:
            pos (np.ndarray): position
            rad (float): radius

        Returns:
            bool: true -> collide
        """
        # within [0, 1]
        if max(pos + rad) > 1 or min(pos - rad) < 0:
            return True
        # check collision
        return self.__collide(get_sphere(pos, rad))

    @stop_watch(verbose=False)
    def __collide_continuous_sphere(
        self, pos1: np.ndarray, pos2: np.ndarray, rad: float
    ) -> bool:
        """private func, check collision with dynamic sphere

        Args:
            pos1 (np.ndarray): 'from' position
            pos2 (np.ndarray): 'to' position
            rad (float): radius

        Returns:
            bool: true -> collide

        Note:
            The following implementation is specialized for
            sphere agents/obstacles. Please use Rodrigues' rotation
            formula when you want to try other cases.
        """
        if len(self.objs) == 0:
            return False

        # assuming that all obstacles are circle
        return any(
            [
                continuous_collide_spheres(
                    pos1, pos2, rad, o.pos, o.pos, o.circumradius
                )
                for o in self.obs
            ]
        )

        # """the following method is based on Rodrigues' rotation formula,
        # c.f.,
        # https://math.stackexchange.com/questions/180418/2672702#2672702
        # """
        # shape = fcl.Capsule(rad, np.linalg.norm(pos1 - pos2))
        # c = (pos1 + pos2) / 2
        # d = np.linalg.norm(pos1 - c)

        # # static case
        # if d == 0:
        #     return self.__collide(get_sphere(pos1, rad))

        # # fast check
        # r = d + rad
        # if all(
        #     [
        #         sum((o.pos - c) ** 2) > (r + o.circumradius) ** 2
        #         for o in self.obs
        #     ]
        # ):
        #     return False

        # # accurate check
        # n = (pos1 - c) / d
        # rot = get_rotation_matrix_3d(np.array([0, 0, 1]), pad(n))
        # tf = fcl.Transform(rot, pad(c))
        # obj = fcl.CollisionObject(shape, tf)
        # return self.__collide(obj)

    def get_dict_for_spars(self) -> list[dict]:
        """used for c++ wrapper (SPARS)"""
        return [o.get_dict_for_spars() for o in self.obs]

    def draw_2d(self, map_size: int) -> np.ndarray:
        """draw 2d image, used in FOV creating

        Args:
            map_size (int): map size

        Returns:
            np.ndarray: 2D array
        """
        if len(self.obs) > 0:
            imgs = np.dstack([x.draw_2d(map_size) for x in self.obs]).max(-1)
        else:
            imgs = np.zeros((map_size, map_size), dtype="float32")

        return imgs


def get_rotation_matrix_3d(frm: np.ndarray, to: np.ndarray) -> np.ndarray:
    """used in the Rodrigues' rotation formula, return rotation matrix

    Args:
        frm (np.ndarray): 'from' position
        to (np.ndarray): 'to' positoin

    Returns:
        np.ndarray: rotation matrix
    """
    if np.array_equal(frm, to):
        return np.identity(3)
    if np.array_equal(frm, -to):
        return -np.identity(3)

    s = frm + to
    return 2.0 * np.outer(s, s) / np.dot(s, s) - np.identity(3)
