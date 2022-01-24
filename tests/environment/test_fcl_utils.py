import numpy as np
import pytest

from ctrm.environment.fcl_utils import (
    collide_boxes,
    collide_spheres,
    continuous_collide_spheres,
    pad,
)


def test_pad():
    res = pad(np.array([1, 1]))
    assert len(res) == 3
    assert res[0] == 1 and res[1] == 1 and res[2] == 0
    res = pad(np.array([1, 1, 1]))
    assert len(res) == 3
    assert res[0] == 1 and res[1] == 1 and res[2] == 1


@pytest.mark.parametrize(
    "pos1, size1, pos2, size2, pos3, size3",
    [
        (  # 2d
            np.array([-1, 0]),
            1,
            np.array([1, 0]),
            1,
            np.array([1.1, 0]),
            1,
        ),
        (  # 3d
            np.array([-1, 0, 0]),
            1,
            np.array([1, 0, 0]),
            1,
            np.array([1.1, 0, 0]),
            1,
        ),
    ],
)
def test_collide_spheres(pos1, size1, pos2, size2, pos3, size3):
    assert collide_spheres(pos1, size1, pos2, size2)
    assert not collide_spheres(pos1, size1, pos3, size3)


@pytest.mark.parametrize(
    "pos1, size1, pos2, size2, pos3, size3",
    [
        (  # 2d
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([1, 0]),
            np.array([1, 1]),
            np.array([1.1, 0]),
            np.array([1, 1]),
        ),
        (  # 3d
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            np.array([1, 0, 0]),
            np.array([1, 1, 1]),
            np.array([1.1, 0, 0]),
            np.array([1, 1, 1]),
        ),
    ],
)
def test_collide_boxes(pos1, size1, pos2, size2, pos3, size3):
    assert collide_boxes(pos1, size1, pos2, size2)
    assert not collide_boxes(pos1, size1, pos3, size3)


@pytest.mark.parametrize(
    "from1, to1, size1, from2, to2, size2, from3, to3, size3",
    [
        (
            np.array([-1, 0]),
            np.array([1, 0]),
            0.1,
            np.array([0, -1]),
            np.array([0, 1]),
            0.1,
            np.array([-1, -1]),
            np.array([-1, 1]),
            0.1,
        ),
        (
            np.array([-1, 0, 0]),
            np.array([1, 0, 0]),
            0.1,
            np.array([0, -1, 0]),
            np.array([0, 1, 0]),
            0.1,
            np.array([0, -1, 1]),
            np.array([0, 1, 1]),
            0.1,
        ),
    ],
)
def test_continuous_collide_spheres(
    from1, to1, size1, from2, to2, size2, from3, to3, size3
):
    assert continuous_collide_spheres(from1, to1, size1, from2, to2, size2)
    assert not continuous_collide_spheres(from1, to1, size1, from3, to3, size3)
