"""dataset class
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import glob
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import tqdm

from ..environment import Instance
from ..planner import Result
from .compiled import get_arr_others_info, get_normed_vec_mag, get_self_info
from .formats import FormatOutput
from .fov import get_approx_cost_to_go_matrix_2d, get_fov2d_occupancy_cost


@dataclass(frozen=True)
class Dataset(torch.utils.data.Dataset):
    """dataset class used for learning process"""

    # basic properties
    datadir: str
    length: int = field(init=False)

    # cache
    on_memory: bool = False
    arr_ins: list[Instance] = field(init=False)
    arr_res: list[Result] = field(init=False)

    # true -> compute input and output
    preprocessing: bool = False

    # for x
    map_size: int = 64
    fov_size: int = 11
    arr_fov: list[torch.Tensor] = field(init=False)  # FOV
    arr_self_info: list[torch.Tensor] = field(init=False)  # self info
    arr_others_info: list[torch.Tensor] = field(init=False)  # others info

    # for y
    format_output: Optional[FormatOutput] = None
    arr_y: list[torch.Tensor] = field(init=False)  # target vector

    # for weight
    use_weight: bool = True
    weight_gamma: float = 50.0
    weight_epsilon: float = 0.0
    arr_w: list[torch.Tensor] = field(init=False)  # weight vector

    def __post_init__(self) -> None:
        """to maintain immutability"""
        original_len = len(glob.glob(os.path.join(self.datadir, "*_ins.pkl")))
        object.__setattr__(self, "length", original_len)

        # load all instances
        if self.on_memory:
            arr_ins: list[Instance] = []
            arr_res: list[Result] = []
            for idx in range(original_len):
                ins, res = self.__getitem__(idx)
                arr_ins.append(ins)
                arr_res.append(res)
            object.__setattr__(self, "arr_ins", arr_ins)
            object.__setattr__(self, "arr_res", arr_res)

        if self.preprocessing:
            self.set_tensors()

    def map_instance_to_tensor(self, idx: int) -> tuple[torch.Tensor, ...]:
        """compute features

        Args:
            idx (int): index

        Returns:
            torch.Tensor: FOVS
                timestep *
                agents *
                (occupancy and cost-to-go)
            torch.Tensor: arr_others_info
                time *
                (self-agent -> other-agent) *
                (current_vec, goal_vec, prev_vec, rad, speed)
            torch.Tensor: arr_self_info
                time *
                agent *
                (goal_vec, prev_vec, rad, max_speed)
            torch.Tensor: y
                time *
                agent *
                (goal_vec)
            torch.Tensor: weight
                time *
                agent *
                weight
        """
        ins, res = self.__getitem__(idx)
        num_agents = ins.num_agents
        T = int(res.maximum_costs)

        #########################
        # compute input

        # compute fov
        occupancy_map = ins.objs.draw_2d(self.map_size)
        cost_to_go = [
            get_approx_cost_to_go_matrix_2d(
                goal=ins.goals[i],
                occupancy_map=occupancy_map,
                rad=ins.rads[i],
            )
            for i in range(ins.num_agents)
        ]
        arr_fov = np.empty((T, ins.num_agents, (self.fov_size ** 2) * 2))
        for t in range(T):
            for j in range(ins.num_agents):
                arr_fov[t, j] = get_fov2d_occupancy_cost(
                    current_pos=res.paths[j][t].pos,
                    occupancy_map=occupancy_map,
                    cost_to_go=cost_to_go[j],
                    fov_size=self.fov_size,
                    flatten=True,
                )
        fovs: torch.Tensor = torch.tensor(arr_fov).float()

        # get relative pos
        arr_current_locs = np.empty((T, ins.num_agents, 2))
        arr_prev_locs = np.empty(arr_current_locs.shape)
        arr_next_locs = np.empty(arr_current_locs.shape)
        goals = np.array(ins.goals)
        for t in range(T):
            for i in range(num_agents):
                arr_current_locs[t, i] = res.paths[i][t].pos
                arr_prev_locs[t, i] = res.paths[i][max(0, t - 1)].pos
                arr_next_locs[t, i] = res.paths[i][t + 1].pos

        # transform other info tensor
        arr_others_info: torch.Tensor = torch.tensor(
            get_arr_others_info(
                arr_current_locs=arr_current_locs,
                goals=goals,
                arr_prev_locs=arr_prev_locs,
                rads=np.array(ins.rads),
                max_speeds=np.array(ins.max_speeds),
            )
        ).float()

        # compute self-info
        arr_self_info: torch.Tensor = get_self_info(
            num_agents=num_agents, arr_others_info=arr_others_info
        )

        #########################
        # compute output

        # compute next-info, goal x next_loc
        arr_next_info = get_normed_vec_mag(
            arr_next_locs.reshape(-1, ins.dim)
            - arr_current_locs.reshape(-1, ins.dim)
        )
        arr_goal_info = get_normed_vec_mag(
            np.repeat(goals.reshape(1, -1, ins.dim), T, axis=0).reshape(
                -1, ins.dim
            )
            - arr_current_locs.reshape(-1, ins.dim)
        )
        arr_sin = np.cross(arr_goal_info[:, :2], arr_next_info[:, :2])
        arr_cos = np.sum(arr_goal_info[:, :2] * arr_next_info[:, :2], axis=1)
        next_infos: torch.Tensor = (
            torch.tensor(arr_next_info).float().reshape(T, num_agents, -1)
        )
        directions_sin: torch.Tensor = (
            torch.tensor(arr_sin).float().reshape(T, num_agents)
        )
        directions_cos: torch.Tensor = (
            torch.tensor(arr_cos).float().reshape(T, num_agents)
        )

        # compute output
        if self.format_output is not None:
            y = self.format_output.get_feature(
                ins=ins,
                res=res,
                arr_next_info=next_infos,
                arr_direction_sin=directions_sin,
                arr_direction_cos=directions_cos,
            )
        else:
            y = torch.empty(0)

        #########################
        # compute weight

        if self.use_weight:
            w = (
                1
                - (
                    torch.exp(
                        -self.weight_gamma
                        * torch.pow(torch.arcsin(directions_sin) / math.pi, 2)
                    )
                )
                + self.weight_epsilon
            )
        else:
            w = torch.ones(*directions_cos.shape)

        return fovs, arr_others_info, arr_self_info, y, w

    def set_tensors(self) -> None:
        """load tensors for all instances"""
        arr_fov = []
        arr_others_info = []
        arr_self_info = []
        arr_y = []
        arr_w = []
        for i in tqdm.tqdm(range(self.__len__()), desc=f"load {self.datadir}"):
            fovs, others_info, self_info, y, w = self.map_instance_to_tensor(i)
            arr_fov.append(fovs)
            arr_others_info.append(others_info)
            arr_self_info.append(self_info)
            arr_y.append(y)
            arr_w.append(w)
        object.__setattr__(self, "arr_fov", arr_fov)
        object.__setattr__(self, "arr_others_info", arr_others_info)
        object.__setattr__(self, "arr_self_info", arr_self_info)
        object.__setattr__(self, "arr_y", arr_y)
        object.__setattr__(self, "arr_w", arr_w)

    def __len__(self) -> int:
        """dataset length"""
        return self.length

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        """get one item

        Returns:
            tuple[Any, ...]: typically,
                Instance: instance
                Result: result
                torch.Tensor: FOV
                torch.Tensor: others info
                torch.Tensor: self info
                torch.Tensor: y (target)
                torch.Tensor: weight
        """
        if self.on_memory and "arr_ins" in dir(self):
            if self.preprocessing and "arr_fov" in dir(self):
                return (
                    self.arr_ins[idx],
                    self.arr_res[idx],
                    self.arr_fov[idx],
                    self.arr_others_info[idx],
                    self.arr_self_info[idx],
                    self.arr_y[idx],
                    self.arr_w[idx],
                )
            else:
                return self.arr_ins[idx], self.arr_res[idx]

        idx_raw_data = int(idx)

        # instance
        with open(
            os.path.join(self.datadir, f"{idx_raw_data:08d}_ins.pkl"), "rb"
        ) as f:
            ins = pickle.load(f)

        # result
        with open(
            os.path.join(self.datadir, f"{idx_raw_data:08d}_res.pkl"), "rb"
        ) as f:
            res = pickle.load(f)

        if self.preprocessing and "arr_fov" in dir(self):
            return (
                ins,
                res,
                self.arr_fov[idx],
                self.arr_others_info[idx],
                self.arr_self_info[idx],
                self.arr_y[idx],
                self.arr_w[idx],
            )
        else:
            return ins, res
