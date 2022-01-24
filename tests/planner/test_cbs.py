from ctrm.planner import CBS
from ctrm.roadmap import get_timed_roadmaps_grid_common


def test_CBS(instance):
    trms = get_timed_roadmaps_grid_common(instance, T=4, size=5)
    planner = CBS(instance, trms)
    res = planner.solve()
    assert res.solved is True
    assert res.name_planner == "CBS"
    assert res.elapsed_planner > 0
    assert res.sum_of_costs > res.maximum_costs > 0
    assert res.sum_of_travel_dists > res.maximum_travel_dists > 0
