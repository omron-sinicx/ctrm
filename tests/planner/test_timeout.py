import numpy as np

from ctrm.environment import Instance, ObstacleSphere
from ctrm.planner import PrioritizedPlanning
from ctrm.roadmap import get_timed_roadmaps_grid_common


def test_planner_timeout():
    ins = Instance(
        2,
        [np.array([0, 0]), np.array([1, 0])],
        [np.array([1, 1]), np.array([0, 1])],
        [0.5, 0.5],
        [0.1, 0.1],
        [0.1, 0.1],
        [ObstacleSphere(pos=np.array([0.5, 0.5]), rad=0.2)],
        2,
    )
    trms = get_timed_roadmaps_grid_common(ins, T=5, size=5)
    planner = PrioritizedPlanning(ins, trms, time_limit=0.001)
    res = planner.solve()
    assert not res.solved
