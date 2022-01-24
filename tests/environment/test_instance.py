import os
import pickle

import numpy as np

from ctrm.environment import (
    Instance,
    ObstacleSphere,
    generate_ins_2d_with_obs_sphere,
    generate_ins_2d_without_obs,
)


def test_generate_ins_random_without_obs():
    num_agents = 2
    dim = 2
    ins = generate_ins_2d_without_obs(
        num_agents=num_agents, max_speed=0.2, rad=0.1,
    )
    assert len(ins.starts) == num_agents
    assert len(ins.goals) == num_agents
    assert all([s.shape[0] == dim for s in ins.starts])
    assert all([g.shape[0] == dim for g in ins.goals])


def test_generate_ins_with_obs_sphere():
    num = 2
    ins = generate_ins_2d_with_obs_sphere(
        num_agents=num,
        max_speed=0.25,
        rad=0.1,
        obs_num=2,
        obs_size_lower_bound=0.05,
        obs_size_upper_bound=0.2,
    )
    assert len(ins.starts) == num
    assert len(ins.goals) == num
    assert all([s.shape[0] == 2 for s in ins.starts])
    assert all([g.shape[0] == 2 for g in ins.goals])


def test_Instance_getstate():
    ins_original = Instance(
        2,
        [np.array([0, 0]), np.array([1, 0])],
        [np.array([1, 1]), np.array([0, 1])],
        [0.5, 0.5],
        [0.1, 0.1],
        [0.1, 0.1],
        [ObstacleSphere(pos=np.array([0.5, 0.5]), rad=0.2)],
        2,
    )

    dirname = "local/fig"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = os.path.join(dirname, "test_ins.pkl")
    with open(filename, "wb") as f:
        pickle.dump(ins_original, f)

    with open(filename, "rb") as f:
        ins_recovery = pickle.load(f)

    assert ins_original.num_agents == ins_recovery.num_agents
