import numpy as np
import pytest

from ctrm.environment import Instance, ObstacleSphere


@pytest.fixture(
    params=[
        (  # basis, 2d
            2,
            [np.array([0.1, 0.1]), np.array([0.9, 0.1])],
            [np.array([0.9, 0.9]), np.array([0.1, 0.9])],
            [0.5, 0.5],
            [0.05, 0.05],
            [0.1, 0.1],
            [],
            2,
        ),
        (  # increase agents
            3,
            [np.array([0.1, 0.1]), np.array([0.9, 0.1]), np.array([0.9, 0.9])],
            [np.array([0.9, 0.9]), np.array([0.1, 0.9]), np.array([0.1, 0.1])],
            [0.5, 0.5, 0.5],
            [0.05, 0.05, 0.05],
            [0.1, 0.1, 0.1],
            [],
            2,
        ),
        (  # 3d
            2,
            [np.array([0.1, 0.1, 0.1]), np.array([0.9, 0.1, 0.1])],
            [np.array([0.9, 0.9, 0.1]), np.array([0.1, 0.9, 0.1])],
            [0.5, 0.5],
            [0.05, 0.05],
            [0.1, 0.1],
            [],
            3,
        ),
        (  # add obstacle
            2,
            [np.array([0.1, 0.1]), np.array([0.9, 0.1])],
            [np.array([0.9, 0.9]), np.array([0.1, 0.9])],
            [0.5, 0.5],
            [0.05, 0.05],
            [0.1, 0.1],
            [ObstacleSphere(pos=np.array([0.5, 0.5]), rad=0.2)],
            2,
        ),
    ]
)
def instance(request):
    return Instance(
        num_agents=request.param[0],
        starts=request.param[1],
        goals=request.param[2],
        max_speeds=request.param[3],
        rads=request.param[4],
        goal_rads=request.param[5],
        obs=request.param[6],
        dim=request.param[7],
    )
