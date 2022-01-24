"""utilities for collision checking
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX

Note:
We used the FLC library at the beginning
so the library includes those wrappers.
However, the latest one does not rely on FCL
because it assumes that all agents and obstacles are 2D circle.
Currently, only "continuous_collide_spheres"
(at the end of this file) is used for collision check.

Ref (FCL):
- Pan, J., Chitta, S., & Manocha, D. (2012, May).
  FCL: A general purpose library for collision and proximity queries.
  In Proc. ICRA (pp. 3859-3866). IEEE.

- https://github.com/flexible-collision-library/fcl
"""

from __future__ import annotations

import fcl
import numpy as np
import sphere_collision_check_wrapper


def pad(pos: np.ndarray) -> np.ndarray:
    """[deprecated] padding position array for
    python-fcl that accounts only for 3d case

    e.g., [1, 1] -> [1, 1, 0] / [1, 1, 1] -> [1, 1, 1]

    Args:
        pos (np.ndarray): 2D or 3D position

    Returns:
        np.ndarray: 3D position
    """
    return pos if pos.shape[0] == 3 else np.pad(pos, 1)[1:4]


def get_sphere(pos: np.ndarray, rad: float) -> fcl.CollisionObject:
    """[deprecated] get sphere FCL object

    Args:
        pos (np.ndarray): position
        rad (float): radius

    Returns:
        fcl.CollisionObject: sphere object
    """
    s = fcl.Sphere(rad)
    t = fcl.Transform(pad(pos))
    return fcl.CollisionObject(s, t)


def get_box(pos: np.ndarray, size: np.ndarray) -> fcl.CollisionObject:
    """[deprecated] get box FCL object

    Args:
        pos (np.ndarray): position
        size (np.ndarray): size array of x, y, (z)

    Returns:
        fcl.CollisionObject: box object
    """
    s = fcl.Box(size[0], size[1], 0 if size.shape[0] == 2 else size[2])
    t = fcl.Transform(pad(pos))
    return fcl.CollisionObject(s, t)


def collide(o1: fcl.CollisionObject, o2: fcl.CollisionObject) -> bool:
    """[deprecated] detect collision of static objects

    Args:
        o1 (fcl.CollisionObject): object 1
        o2 (fcl.CollisionObject): object 2

    Returns:
        bool: true -> collide
    """
    req = fcl.CollisionRequest()
    res = fcl.CollisionResult()
    fcl.collide(o1, o2, req, res)
    return res.is_collision


def collide_spheres(
    pos1: np.ndarray, size1: float, pos2: np.ndarray, size2: float
) -> bool:
    """[deprecated] detect collision of two spheres

    Args:
        pos1 (np.ndarray): position of object 1
        size1 (float): radius of object 1
        pos2 (np.ndarray): position of object 2
        size2 (float): radius of object 2

    Returns:
        bool: true -> collide
    """
    o1 = get_sphere(pos1, size1)
    o2 = get_sphere(pos2, size2)
    return collide(o1, o2)


def collide_boxes(
    pos1: np.ndarray, size1: np.ndarray, pos2: np.ndarray, size2: np.ndarray
) -> bool:
    """[deprecated] detect collision of two boxes

    Args:
        pos1 (np.ndarray): position of object 1
        size1 (np.ndarray): size (x, y, (z)) of object 1
        pos2 (np.ndarray): position of object 2
        size2 (np.ndarray): size (x, y, (z)) of object 2

    Returns:
        bool: true -> collide
    """
    o1 = get_box(pos1, size1)
    o2 = get_box(pos2, size2)
    return collide(o1, o2)


def continuous_collide(
    o1: fcl.CollisionObject,
    t1_f: np.ndarray,
    o2: fcl.CollisionObject,
    t2_f: np.ndarray,
) -> bool:
    """[deprecated] detect collision of dynamic objects

    Args:
        o1 (fcl.CollisionObject): object 1
        t1_f (np.ndarray): end position of object 1
        o2 (fcl.CollisionObject): object 2
        t1_f (np.ndarray): end position of object 2

    Returns:
        bool: true -> collide
    """
    req = fcl.ContinuousCollisionRequest()
    res = fcl.ContinuousCollisionResult()
    fcl.continuousCollide(o1, t1_f, o2, t2_f, req, res)
    return res.is_collide


def continuous_collide_dynamic_and_static_obj(
    o1: fcl.CollisionObject, t1_f: np.ndarray, o2: fcl.CollisionObject,
) -> bool:
    """[deprecated] detect collision of dynamic object and static object

    Args:
        o1 (fcl.CollisionObject): dynamic object
        t1_f (np.ndarray): end position of dynamic object
        o2 (fcl.CollisionObject): static object

    Returns:
        bool: true -> collide
    """
    return continuous_collide(o1, t1_f, o2, o2.getTransform())


def continuous_collide_sphere_static_obj(
    pos1: np.ndarray, pos2: np.ndarray, rad: float, obj: fcl.CollisionObject,
) -> bool:
    """[deprecated] detect collision between dynamic sphere and static object

    Args:
        pos1 (np.ndarray): 'from' position of sphere
        pos2 (np.ndarray): 'to' position of sphere
        rad (float): radius
        obj (fcl.CollisionObject): static object

    Returns:
        bool: true -> collide
    """
    o = get_sphere(pos1, rad)
    t_f = fcl.Transform(pad(pos2))
    return continuous_collide_dynamic_and_static_obj(o, t_f, obj)


def continuous_collide_spheres(
    from1: np.ndarray,
    to1: np.ndarray,
    rad1: float,
    from2: np.ndarray,
    to2: np.ndarray,
    rad2: float,
) -> bool:
    """detect collision between two dynamic spheres

    Note:
        The current implementation does not use FCL.

    Args:
        from1 (np.ndarray): 'from' position of sphere 1
        to1 (np.ndarray): 'to' position of sphere 1
        rad1 (float): radius of sphere 1
        from2 (np.ndarray): 'from' position of sphere 2
        to2 (np.ndarray): 'to' position of sphere 2
        rad2 (float): radius of sphere 2

    Returns:
        bool: true -> collide
    """
    # use cpp helper
    return sphere_collision_check_wrapper.continuousCollideSpheres(
        from1, to1, rad1, from2, to2, rad2
    )

    """FCL example"""
    # o1 = get_sphere(from1, rad1)
    # t1_f = fcl.Transform(pad(to1))
    # o2 = get_sphere(from2, rad2)
    # t2_f = fcl.Transform(pad(to2))
    # return continuous_collide(o1, t1_f, o2, t2_f)

    """without FCL, this is much faster -> coded in cpp_wrapper"""
    # def dist_pow2(t):
    #     return sum(
    #         (((1 - t) * from1 + t * to1) - ((1 - t) * from2 + t * to2)) ** 2
    #     )

    # corr = -from1 + to1 + from2 - to2
    # a = np.sum(corr ** 2)
    # b = np.dot(corr, from1 - from2)

    # print(a, b)

    # if b >= 0:  # two agents are going away
    #     min_dist_pow2 = dist_pow2(0)
    # elif a + b <= 0:  # two agents are approaching
    #     min_dist_pow2 = dist_pow2(1)
    # else:  # otherwise
    #     min_dist_pow2 = dist_pow2(-b / a)

    # return min_dist_pow2 <= (rad1 + rad2) ** 2
