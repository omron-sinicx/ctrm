import numpy as np

from ctrm.environment import Instance, ObstacleSphere
from ctrm.planner import PrioritizedPlanning
from ctrm.roadmap import get_timed_roadamaps_SPARS_2d_common


def test_spars_prioritized_planning():
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
    trms = get_timed_roadamaps_SPARS_2d_common(
        ins, T=5, sparse_delta_fraction=0.2, time_limit_sec=2.0
    )
    planner = PrioritizedPlanning(ins, trms)
    res = planner.solve()
    assert res.solved
    assert res.elapsed_planner > 0
    assert res.sum_of_costs > res.maximum_costs > 0
    assert res.sum_of_travel_dists > res.maximum_travel_dists > 0
