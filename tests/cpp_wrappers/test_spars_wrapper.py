import os

import matplotlib.pyplot as plt
import numpy as np
from spars_wrapper import getSparsRoadmap2d

from ctrm.environment import ObstacleBox, ObstacleSphere


def test_getSparsRoadmap2d():
    obs = [
        ObstacleSphere(pos=np.array([0.3, 0.3]), rad=0.1).get_dict_for_spars(),
        ObstacleBox(
            pos=np.array([0.7, 0.7]), size=np.array([0.01, 0.2])
        ).get_dict_for_spars(),
    ]

    # agent
    rad = 0.1
    speed = 0.2
    sparse_delta_fraction = 0.1
    dense_delta_fraction = 0.1
    stretch_factor = 1.3
    max_sample_num = 100000
    time_limit = 0.1
    lower_bound = 0
    upper_bound = 1

    # get roadmap
    res = getSparsRoadmap2d(
        rad,
        speed,
        sparse_delta_fraction,
        dense_delta_fraction,
        stretch_factor,
        max_sample_num,
        time_limit,
        obs,
        lower_bound,
        upper_bound,
    )
    rmp = res[0]

    for s in rmp:
        pos1 = np.array([s[0], s[1]])
        for i in s[-1]:
            _s = rmp[i]
            pos2 = np.array([_s[0], _s[1]])
            assert np.linalg.norm(pos1 - pos2) <= speed

    # illustrate roadmap
    plt.figure(figsize=(4, 4))
    for v in rmp:
        plt.scatter([v[0]], [v[1]], marker="x", color="blue")
        for j in v[-1]:
            plt.plot(
                [v[0], rmp[j][0]],
                [v[1], rmp[j][1]],
                color="black",
                linewidth=0.2,
            )

    # save result
    dirname = "local/fig"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = os.path.join(dirname, "test_getSparsRoadmap2d.png")
    plt.savefig(
        filename, pad_inches=0.05, transparent=False, bbox_inches="tight"
    )
    print(f"file is saved into {dirname}")
