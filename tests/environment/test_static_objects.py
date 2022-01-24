import numpy as np
import pytest

from ctrm.environment import ObstacleSphere, StaticObjects


@pytest.fixture
def obstacles2d():
    rad = float(np.sqrt(2) / 4)
    return {
        "rad": rad,
        "obs": [
            ObstacleSphere(np.array([1, 0]), rad),
            ObstacleSphere(np.array([0, 1]), rad),
        ],
    }


def test_StaticObjects_2d(obstacles2d):
    rad = obstacles2d["rad"]
    obj = StaticObjects(obstacles2d["obs"])

    # static
    assert obj.collide_sphere(np.array([0.5, 0.5]), rad)
    assert not obj.collide_sphere(np.array([0.5, 0.5]), rad - 0.01)

    # dynamic
    s = np.array([0, 0])
    g = np.array([1, 1])
    assert obj.collide_continuous_sphere(s, g, rad)
    assert not obj.collide_continuous_sphere(s, g, rad - 0.01)


@pytest.fixture
def obstacles3d():
    rad = 0.25
    return {
        "rad": rad,
        "obs": [
            ObstacleSphere(np.array([0.5, 0.5, 0]), rad),
            ObstacleSphere(np.array([0.5, 0.5, 1]), rad),
        ],
    }


def test_StaticObjects_3d(obstacles3d):
    rad = obstacles3d["rad"]
    obj = StaticObjects(obstacles3d["obs"])

    # static
    assert obj.collide_sphere(np.array([0.5, 0.5, 0.5]), rad)
    assert not obj.collide_sphere(np.array([0.5, 0.5, 0.5]), rad - 0.0005)

    # dynamic
    s = np.array([0, 0, 0.5])
    g = np.array([1, 1, 0.5])
    assert obj.collide_continuous_sphere(s, g, rad)
    assert not obj.collide_continuous_sphere(s, g, rad - 0.0005)
