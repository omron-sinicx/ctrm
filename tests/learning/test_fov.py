import numpy as np
import pytest

from ctrm.environment import Instance, ObstacleSphere
from ctrm.learning.fov import (
    get_approx_cost_to_go_matrix_2d,
    get_fov2d_occupancy_cost,
    get_map_coord_2d,
)


@pytest.fixture
def ins():
    return Instance(
        num_agents=1,
        starts=[np.array([0.0, 0.0])],
        goals=[np.array([1.0, 1.0])],
        max_speeds=[0.1],
        rads=[0.05],
        goal_rads=[0.01],
        obs=[
            ObstacleSphere(np.array([0.5, 0.5]), 0.2),
            ObstacleSphere(np.array([1, 0]), 0.2),
        ],
        dim=2,
    )


def test_get_approx_cost_to_go(ins):
    map_size = 16

    occupancy_map = ins.objs.draw_2d(map_size)
    goal = ins.goals[0]
    cost_to_go_map = get_approx_cost_to_go_matrix_2d(
        goal, occupancy_map, ins.rads[0]
    )

    assert cost_to_go_map[get_map_coord_2d(ins.starts[0], map_size)] == 30


def test_get_fov2d_occupancy_cost(ins):
    map_size = 16
    fov_size = 5
    goal = ins.goals[0]
    occupancy_map = ins.objs.draw_2d(map_size)
    cost_to_go = get_approx_cost_to_go_matrix_2d(
        goal, occupancy_map, ins.rads[0]
    )

    fov1 = get_fov2d_occupancy_cost(
        np.array([0, 0]), occupancy_map, cost_to_go, fov_size
    )
    expected_fov1 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
        ]
    )
    assert np.array_equal(fov1, expected_fov1)

    fov2 = get_fov2d_occupancy_cost(
        np.array([0, 1]), occupancy_map, cost_to_go, fov_size
    )
    expected_fov2 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
            ],
        ]
    )
    assert np.array_equal(fov2, expected_fov2)
