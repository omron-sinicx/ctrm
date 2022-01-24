import numpy as np

from ctrm.environment import Instance
from ctrm.planner import PrioritizedPlanning
from ctrm.roadmap import get_timed_roadmaps_grid_common


def test_PrioritizedPlanning(instance):
    trms = get_timed_roadmaps_grid_common(instance, T=4, size=5)
    planner = PrioritizedPlanning(instance, trms)
    res = planner.solve()
    assert res.solved
    assert res.name_planner == "PrioritizedPlanning"
    assert res.elapsed_planner > 0
    assert res.sum_of_costs > res.maximum_costs > 0
    assert res.sum_of_travel_dists > res.maximum_travel_dists > 0


def test_PrioritizedPlanning_no_collision():
    ins = Instance(
        2,
        [np.array([0, 0.5]), np.array([1, 0.5])],
        [np.array([1, 0.5]), np.array([0, 0.5])],
        [0.5, 0.5],
        [0.1, 0.1],
        [0.1, 0.1],
        [],
        2,
    )

    trms = get_timed_roadmaps_grid_common(ins, T=4, size=5)
    planner = PrioritizedPlanning(ins, trms, agent_collision=False)
    res = planner.solve()
    assert res.solved
    assert res.maximum_costs == 2
