"""generation of CTRMs
Author: Keisuke Okumura / Ryo Yonetani
Affiliation: TokyoTech & OSX / OSX
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..environment import Instance
from ..learning import reconstruct
from ..roadmap import TimedRoadmap
from ..roadmap.utils import valid_move
from .utils import append_goals, format_trms, merge_samples


def get_timed_roadmaps_multiple_paths_with_learned_indicator(
    ins: Instance,
    pred_basename: str,
    prob_uniform_sampling_after_goal: float = 0.9,
    prob_uniform_bias: float = 0.0,
    prob_uniform_gamma: float = 5.0,
    max_T: int = 64,
    N_traj: int = 25,
    max_attempt: int = 3,
    randomize_indicator: bool = False,
    merge_distance_rate: float = 0.25,
    verbose: int = 0,
) -> list[TimedRoadmap]:
    """
    Args:
        ins (Instance): instance
        pred_basename (str): model directory + label, e.g., xxxx/best
        prob_uniform_sampling_after_goal (:obj:`float`, optional):
            random walk parameter after arrival at goal
        prob_uniform_bias (:obj:`float`, optional): random walk parameter
        prob_uniform_gamma (:obj:`float`, optional): random walk parameter
        max_T (:obj:`int`, optional): maximum timestep
        N_traj (:obj:`int`, optional): number of repetition of generating paths
        max_attempt (:obj:`int`, optional): maximum retry number of random walk
        randomize_indicator (:obj:`bool`, optional):
            if indicators are also randomized
        merge_distance_rate (:obj:`float`, optional):
            distance to judge compatibility
        verbose (:obj:`int`, optional): >0 -> print additional info

    Returns:
        list[TimedRoadmap]: timed roadmaps
    """
    # reconstruct model
    pred_model, format_input, format_output = reconstruct(pred_basename)
    format_input.set_instance(ins)
    dim_ind = format_output.get_dim_indicators()

    # initialize
    trms: list[TimedRoadmap] = [
        TimedRoadmap(ins.starts[i]) for i in range(ins.num_agents)
    ]

    makespan = None

    # trajectory
    for k_traj in tqdm(
        range(N_traj), disable=(verbose == 0), desc="roadmap_gen"
    ):
        arr_loc_next = []  # pos at t+1
        arr_loc_current = ins.starts  # pos at t
        arr_loc_prev = ins.starts  # pos at t-1
        arrived_at_goals = [
            False
        ] * ins.num_agents  # true -> goal, false -> not yet

        for t in range(1, max_T + 1):  # timestep
            with torch.no_grad():
                X = format_input.get_feature_one_step(
                    ins=ins,
                    arr_current_pos=arr_loc_current,
                    arr_prev_pos=arr_loc_prev,
                )
                X = torch.cat([X] * dim_ind)

                prob_random_walk = (
                    1
                    - np.exp(
                        -prob_uniform_gamma
                        * t
                        / (max_T if makespan is None else makespan)
                    )
                ) * (1 - prob_uniform_bias) + prob_uniform_bias

                if randomize_indicator & (np.random.rand() > prob_random_walk):
                    IND = torch.nn.functional.one_hot(
                        torch.randint(0, dim_ind, (ins.num_agents * dim_ind,)),
                        num_classes=dim_ind,
                    )
                else:
                    IND = torch.nn.functional.one_hot(
                        torch.argmax(
                            pred_model.indicator(X), -1  # type: ignore
                        ),
                        dim_ind,
                    )

                # for resampling
                SAMPLES = (
                    pred_model.sample(  # type: ignore
                        X, IND
                    )
                    .detach()
                    .numpy()
                )
            arr_loc_current_repeat = np.repeat(
                np.expand_dims(np.array(arr_loc_current), 0),
                int(X.shape[0] / ins.num_agents),
                axis=0,
            ).reshape(-1, 2)
            Y = arr_loc_current_repeat + SAMPLES[:, :2] * SAMPLES[
                :, 2
            ].reshape(-1, 1)

            for i in range(ins.num_agents):  # agent
                loc_current = arr_loc_current[i]

                # define valid moves
                def valid_edge(pos1: np.ndarray, pos2: np.ndarray) -> bool:
                    if min(pos2) < ins.rads[i] or 1 - ins.rads[i] < max(pos2):
                        return False
                    return valid_move(
                        pos1, pos2, ins.max_speeds[i], ins.rads[i], ins.objs
                    )

                # random walk
                def sample_uniform_i():
                    for _ in range(max_attempt):
                        mag = ins.max_speeds[i] * np.random.rand()
                        theta = np.random.rand() * np.pi * 2
                        loc = (
                            np.array([np.sin(theta), np.cos(theta)]) * mag
                            + loc_current
                        )
                        if valid_edge(loc_current, loc):
                            return loc
                    return loc_current

                # get sample
                if (
                    (
                        not arrived_at_goals[i]
                        and np.random.rand() < prob_random_walk
                    )
                    or (
                        arrived_at_goals[i]
                        and np.random.rand() > prob_uniform_sampling_after_goal
                    )
                    or (k_traj < dim_ind)
                ):
                    is_valid_sample = False
                    for k in range(int(X.shape[0] / ins.num_agents)):
                        loc_pred = Y[i + k * ins.num_agents]
                        if valid_edge(loc_current, loc_pred):
                            is_valid_sample = True
                            break
                    if not is_valid_sample:
                        loc_pred = sample_uniform_i()
                else:
                    loc_pred = sample_uniform_i()

                # check merge
                loc_next = merge_samples(
                    loc=loc_pred,
                    t=t,
                    agent=i,
                    ins=ins,
                    trm=trms[i],
                    merge_distance=merge_distance_rate * ins.max_speeds[i],
                )
                arr_loc_next.append(loc_next)

                # check goal condition
                if valid_edge(loc_next, ins.goals[i]):
                    arrived_at_goals[i] = True

            # check goal cond
            if all(arrived_at_goals):
                if makespan is not None:
                    makespan = max(makespan, t)
                else:
                    makespan = t
                break

            # update loc_prev
            arr_loc_prev = arr_loc_current
            arr_loc_current = arr_loc_next
            arr_loc_next = []

    # align length of roadmaps for planners
    format_trms(ins, trms)

    # append goals
    append_goals(ins, trms)

    return trms
