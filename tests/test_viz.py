import os

import numpy as np

from ctrm.environment import Instance, ObstacleBox, ObstacleSphere
from ctrm.planner import PrioritizedPlanning
from ctrm.roadmap import (
    get_timed_roadamaps_SPARS_2d_common,
    get_timed_roadmaps_grid_common,
    get_timed_roadmaps_random_common,
)
from ctrm.viz import plot_trm_onestep, simple_plot_2d


def test_plot_2d():
    ins = Instance(
        2,
        [np.array([0, 0]), np.array([1, 0])],
        [np.array([1, 1]), np.array([0, 1])],
        [0.5, 0.5],
        [0.1, 0.1],
        [0.1, 0.1],
        [
            ObstacleSphere(pos=np.array([0.5, 0.5]), rad=0.2),
            ObstacleBox(pos=np.array([0.9, 0.5]), size=np.array([0.2, 0.2])),
        ],
        2,
    )

    trms = get_timed_roadmaps_grid_common(ins, T=4, size=5)
    planner = PrioritizedPlanning(ins, trms)
    result = planner.solve()

    dirname = "local/fig"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    simple_plot_2d(
        ins, filename=os.path.join(dirname, "test_simple_plot_2d_ins-only.png")
    )
    simple_plot_2d(
        ins, result.paths, os.path.join(dirname, "test_simple_plot_2d.png")
    )
    print(f"file is saved into {dirname}")
    assert True


def test_plot_trm_onestep():
    ins = Instance(
        2,
        [np.array([0, 0]), np.array([1, 0])],
        [np.array([1, 1]), np.array([0, 1])],
        [0.5, 0.5],
        [0.1, 0.1],
        [0.1, 0.1],
        [
            ObstacleSphere(pos=np.array([0.5, 0.5]), rad=0.2),
            ObstacleBox(pos=np.array([0.9, 0.5]), size=np.array([0.2, 0.2])),
        ],
        2,
    )
    dirname = "local/fig"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    trms = get_timed_roadmaps_grid_common(ins, T=2, size=5)
    plot_trm_onestep(
        trms[0], ins, os.path.join(dirname, "test_plot_trm_onestep_grid.png")
    )

    trms = get_timed_roadmaps_random_common(ins, T=2, num=30)
    plot_trm_onestep(
        trms[0], ins, os.path.join(dirname, "test_plot_trm_onestep_random.png")
    )

    trms = get_timed_roadamaps_SPARS_2d_common(ins, T=2, time_limit_sec=0.1)
    plot_trm_onestep(
        trms[0], ins, os.path.join(dirname, "test_plot_trm_onestep_spars.png")
    )
