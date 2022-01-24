"""definition of problem instance
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .obstacle import Obstacle, ObstacleSphere
from .static_objects import StaticObjects


@dataclass(frozen=True)
class Instance:
    """definition of problem instance"""

    num_agents: int  # number of agents
    starts: list[np.ndarray]  # pos * num_agents, pos: np.ndarray
    goals: list[np.ndarray]  # pos * num_agents, pos: np.ndarray
    max_speeds: list[float]  # scalar * num_agents
    rads: list[float]  # list of radius, assuming sphere
    goal_rads: list[float]  # list of goal radius, assuming sphere
    obs: list[Obstacle] = field(default_factory=list)  # user defined obstacles
    dim: int = 2  # dimension
    objs: StaticObjects = field(
        init=False
    )  # static objects, set automatically

    def __post_init__(self):
        """to maintain immutability"""
        object.__setattr__(self, "objs", StaticObjects(self.obs))

    def __getstate__(self):
        """called when pickling"""
        state = self.__dict__.copy()
        del state["objs"]
        return state

    def __setstate__(self, state):
        """called when un-pickling"""
        self.__dict__.update(state)
        self.__post_init__()


# below: several functions that generate instances automatically


def generate_ins_2d_with_obs_hetero(
    num_agents: int,
    max_speeds_cands: Union[list[float], str],
    rads_cands: Union[list[float], str],
    obs_num: int,
    obs_size_lower_bound: float = 0.05,
    obs_size_upper_bound: float = 0.2,
    min_dist: Optional[float] = None,
) -> Instance:
    """generate one instance with heterogeneous agents and sphere obstacles

    Args:
        num_agents (int): number of agents
        max_speeds_cands (Union[list[float], str]): candidates of max speeds
        rads_cands (Union[list[float], str]): candidates of radius
        obs_num (int): number of obstacles
        obs_size_lower_bound (:obj:`float`, optional): size of obstacles
        obs_size_upper_bound (:obj:`float`, optional): size of obstacles
        min_dist (:obj:`Optional[float]`): minimum distance between agents

    Returns:
        Instance: instance
    """

    # accept string with comma
    if isinstance(rads_cands, str):
        rads_cands = [float(i) for i in rads_cands.split(",")]
    if isinstance(max_speeds_cands, str):
        max_speeds_cands = [float(i) for i in max_speeds_cands.split(",")]

    obs: list[ObstacleSphere] = [
        ObstacleSphere(
            pos=np.random.rand(2),
            rad=np.random.rand()
            * (obs_size_upper_bound - obs_size_lower_bound)
            / 2
            + obs_size_lower_bound / 2,
        )
        for _ in range(obs_num)
    ]

    # set radius and max_speeds
    rads: list[float] = [
        rads_cands[i] for i in np.random.choice(len(rads_cands), num_agents)
    ]
    max_speeds: list[float] = [
        max_speeds_cands[i]
        for i in np.random.choice(len(max_speeds_cands), num_agents)
    ]

    # set starts
    starts: list[np.ndarray] = []
    while len(starts) < num_agents:
        rad = rads[len(starts)]
        pos = np.random.rand(2) * (1 - 2 * rad) + rad
        # check collisions with obstacles
        if any([np.linalg.norm(pos - o.pos) <= rad + o.rad for o in obs]):
            continue
        # check collisions with other starts
        if any(
            [
                np.linalg.norm(pos - starts[i]) <= rad + rads[i]
                for i in range(len(starts))
            ]
        ):
            continue
        starts.append(pos)

    # set goals
    goals: list[np.ndarray] = []
    while len(goals) < num_agents:
        rad = rads[len(goals)]
        pos = np.random.rand(2) * (1 - 2 * rad) + rad
        if (
            min_dist is not None
            and np.linalg.norm(pos - starts[len(goals)]) < min_dist
        ):
            continue
        if any([np.linalg.norm(pos - o.pos) <= rad + o.rad for o in obs]):
            continue
        # check collisions with other goals
        if any(
            [
                np.linalg.norm(pos - goals[i]) <= rad + rads[i]
                for i in range(len(goals))
            ]
        ):
            continue
        goals.append(pos)

    return Instance(
        num_agents=num_agents,
        starts=starts,
        goals=goals,
        max_speeds=max_speeds,
        rads=rads,
        goal_rads=[0.01] * num_agents,
        obs=obs,  # type: ignore
        dim=2,
    )


def generate_ins_2d_with_obs_hetero_nonfix_agents(
    num_agents_min: int,
    num_agents_max: int,
    max_speeds_cands: Union[list[float], str],
    rads_cands: Union[list[float], str],
    obs_num: int,
    obs_size_lower_bound: float = 0.05,
    obs_size_upper_bound: float = 0.2,
    min_dist: Optional[float] = None,
) -> Instance:
    """generate one instance with various number of
    heterogeneous agents and sphere obstacles

    Note:
        For details, check generate_ins_2d_with_obs_hetero
    """

    num_agents = np.random.randint(num_agents_min, num_agents_max)
    return generate_ins_2d_with_obs_hetero(
        num_agents=num_agents,
        max_speeds_cands=max_speeds_cands,
        rads_cands=rads_cands,
        obs_num=obs_num,
        obs_size_lower_bound=obs_size_lower_bound,
        obs_size_upper_bound=obs_size_upper_bound,
        min_dist=min_dist,
    )


def generate_ins_2d_with_obs_sphere(
    num_agents: int,
    max_speed: float,
    rad: float,
    obs_num: int,
    obs_size_lower_bound: float = 0.05,
    obs_size_upper_bound: float = 0.2,
    min_dist: Optional[float] = None,
) -> Instance:
    """generate one instance with fixed number of
    homogeneous agents and sphere obstacles

    Note:
        For details, check generate_ins_2d_with_obs_hetero
    """

    return generate_ins_2d_with_obs_hetero(
        num_agents=num_agents,
        max_speeds_cands=[max_speed],
        rads_cands=[rad],
        obs_num=obs_num,
        obs_size_lower_bound=obs_size_lower_bound,
        obs_size_upper_bound=obs_size_upper_bound,
        min_dist=min_dist,
    )


def generate_ins_2d_with_obs_sphere_nonfix_agents(
    num_agents_min: int,
    num_agents_max: int,
    max_speed: float,
    rad: float,
    obs_num: int,
    obs_size_lower_bound: float = 0.05,
    obs_size_upper_bound: float = 0.2,
    min_dist: Optional[float] = None,
) -> Instance:
    """generate one instance with various number of
    homogeneous agents and sphere obstacles

    Note:
        For details, check generate_ins_2d_with_obs_hetero
    """

    num_agents = np.random.randint(num_agents_min, num_agents_max)
    return generate_ins_2d_with_obs_sphere(
        num_agents=num_agents,
        max_speed=max_speed,
        rad=rad,
        obs_num=obs_num,
        obs_size_lower_bound=obs_size_lower_bound,
        obs_size_upper_bound=obs_size_upper_bound,
        min_dist=min_dist,
    )


def generate_ins_2d_without_obs(
    num_agents: int,
    max_speed: float,
    rad: float,
    min_dist: Optional[float] = None,
) -> Instance:
    """generate one instance with fixed number of
    homogeneous agents without obstacles

    Note:
        For details, check generate_ins_2d_with_obs_hetero
    """

    return generate_ins_2d_with_obs_sphere(
        num_agents=num_agents,
        max_speed=max_speed,
        rad=rad,
        obs_num=0,
        obs_size_lower_bound=0,
        obs_size_upper_bound=0,
        min_dist=min_dist,
    )


def generate_ins_2d_cross(
    num_agents: int,
    max_speed: float,
    rad: float,
    set_obs: bool = False,
    noise: float = 0,
    rotation: bool = False,
    dispersion: Union[float, list[float]] = 0.45,
    **kwargs
) -> Instance:
    """deprecated: generate symmetry breaking instance

    Args:
        num_agents (int): number of agents, <= 4
        max_speed (float): max_speed
        rad (float): radius
        set_obs (:obj:`bool`, optional): set one obstacle at the center
        noise (:obj:`float`, optional):
            set uniform noise for agents' starts and goals
        rotation (:obj:`bool`): rotate configuration
        dispersion (:obj:`Union[float, list[float]]`):
            initial distance between agents

    Returns:
        Instance: instance
    """

    assert num_agents <= 4

    d = (
        dispersion
        if isinstance(dispersion, float)
        else np.random.choice(dispersion)
    )
    assert d > rad, "collide"

    start_goal_cands = np.array(
        [[-d, -d, d, d], [d, d, -d, -d], [d, -d, -d, d], [-d, d, d, -d],]
    )

    # rotate
    if rotation:
        t = np.random.rand() * np.pi * 2
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        start_goal_cands = np.dot(R, start_goal_cands.reshape(2, -1)).reshape(
            4, -1
        )

    # noise
    if noise > 0:
        start_goal_cands = (
            start_goal_cands + np.random.rand(4, 4) * noise - noise / 2
        )

    np.random.shuffle(start_goal_cands)
    start_goal_cands += 0.5
    start_goal_cands = np.clip(start_goal_cands, rad, 1 - rad)  # type: ignore

    starts = list(start_goal_cands[:num_agents, :2])
    goals = list(start_goal_cands[:num_agents, 2:])
    obs = (
        []
        if set_obs is False
        else [ObstacleSphere(pos=np.array([0.5, 0.5]), rad=0.2)]
    )

    return Instance(
        num_agents=num_agents,
        starts=starts,
        goals=goals,
        max_speeds=[max_speed] * num_agents,
        rads=[rad] * num_agents,
        goal_rads=[0.01] * num_agents,
        obs=obs,  # type: ignore
        dim=2,
    )
