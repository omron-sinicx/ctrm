import numpy as np
import pytest

from ctrm.environment import Instance, ObstacleSphere
from ctrm.roadmap import (
    get_timed_roadmaps_grid,
    get_timed_roadmaps_grid_common,
)


@pytest.fixture
def ins():
    return Instance(
        2,
        [np.array([0, 0]), np.array([1, 0])],
        [np.array([1, 1]), np.array([0, 1])],
        [0.5, 0.5],
        [0.1, 0.1],
        [0.1, 0.1],
        [ObstacleSphere(pos=np.array([0.5, 0.5]), rad=0.2)],
        2,
    )


def test_get_timed_roadmaps_random(ins):
    kwargs = {"ins": ins, "T": 3, "size": 5}
    assert len(get_timed_roadmaps_grid(**kwargs)) == ins.num_agents
    assert len(get_timed_roadmaps_grid_common(**kwargs)) == ins.num_agents
