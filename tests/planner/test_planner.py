import numpy as np

from ctrm.planner import get_cost, get_travel_dist
from ctrm.roadmap import TimedNode


def test_get_cost():
    goal = np.array([1, 0])
    rad = 0.1

    path1 = [
        TimedNode(0, 0, np.array([0.0, 0])),
        TimedNode(1, 0, np.array([0.8, 0])),
        TimedNode(2, 0, np.array([0.9, 0])),
        TimedNode(3, 0, np.array([1.0, 0])),
    ]
    assert get_cost(path1, goal, rad) == 2

    path2 = [
        TimedNode(0, 0, np.array([0.0, 0])),
        TimedNode(1, 0, np.array([0.8, 0])),
        TimedNode(2, 0, np.array([0.9, 0])),
    ]
    assert get_cost(path2, goal, rad) == 2

    path3 = [
        TimedNode(0, 0, np.array([0.0, 0])),
        TimedNode(1, 0, np.array([0.9, 0])),
        TimedNode(2, 0, np.array([0.0, 0])),
        TimedNode(3, 0, np.array([1.0, 0])),
    ]
    assert get_cost(path3, goal, rad) == 3


def test_get_travel_dist():
    path1 = [
        TimedNode(0, 0, np.array([0.0, 0])),
        TimedNode(1, 0, np.array([0.8, 0])),
        TimedNode(2, 0, np.array([0.9, 0])),
        TimedNode(3, 0, np.array([1.0, 0])),
    ]
    assert get_travel_dist(path1) == 1.0

    path2 = [
        TimedNode(0, 0, np.array([0.0, 0])),
        TimedNode(1, 0, np.array([0.8, 0])),
        TimedNode(2, 0, np.array([0.9, 0])),
    ]
    assert get_travel_dist(path2) == 0.9

    path3 = [
        TimedNode(0, 0, np.array([0.0, 0])),
        TimedNode(1, 0, np.array([0.9, 0])),
        TimedNode(2, 0, np.array([0.0, 0])),
        TimedNode(3, 0, np.array([1.0, 0])),
    ]
    assert get_travel_dist(path3) == 2.8
