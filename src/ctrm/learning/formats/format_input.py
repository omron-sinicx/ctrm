"""formatting the input feature
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import pickle
from abc import abstractmethod

import numpy as np
import torch

from ...environment import Instance
from ...planner import Result
from ...utils import get_device_name
from ..compiled import get_arr_others_info, get_self_info
from ..fov import get_approx_cost_to_go_matrix_2d, get_fov2d_occupancy_cost
from ..model import (AgentEncoder, FOVEncoder, Net, reconstruct_agent_encoder,
                     reconstruct_fov_encoder)
from .base import Format
from .utils import get_goalvec_mag


class FormatInput(Format):
    """input format"""

    succ_format_name: str = "_format_input.pkl"

    def set_instance(self, ins: Instance) -> None:
        pass

    def get_trainees(self) -> list[Net]:
        """return models, e.g., agent-encoder"""
        return []

    def reconstruct_trainees(self, basename: str) -> None:
        """reconstruct neural networks"""
        pass

    @abstractmethod
    def get_feature(
        self,
        ins: Instance,
        res: Result,
        input_tensor_fov: torch.Tensor,
        arr_others_info: torch.Tensor,
        arr_self_info: torch.Tensor,
    ) -> torch.Tensor:
        """used with learning process"""
        pass

    @abstractmethod
    def get_feature_one_step(
        self,
        ins: Instance,
        arr_current_pos: list[np.ndarray],
        arr_prev_pos: list[np.ndarray],
    ) -> torch.Tensor:
        """used with inference phase"""
        pass

    def get_fov_size(self) -> int:
        return 0

    def get_map_size(self) -> int:
        return 0


def reconstruct_format_input(basename) -> FormatInput:
    """reconstruct input format from pickle file

    Args:
        basename (str): dirname + label, e.g., xxxx/best

    Returns:
        FormatInput: formatter (with ML-model)
    """
    with open(basename + FormatInput.succ_format_name, "rb") as fb:
        format_input = pickle.load(fb)
        format_input.reconstruct_trainees(basename)
        return format_input


class Format2D_GoalVec(FormatInput):
    """toy example"""

    def __init__(self, **kwargs) -> None:
        super().__init__(dim=3)

    def __repr__(self) -> str:
        return "(goal_vec)"

    def __call__(self, **kwargs) -> torch.Tensor:
        return torch.Tensor(get_goalvec_mag(**kwargs)).float()


class Format2D_CTRM_Input(FormatInput):
    """input feature of CTRMNet

    Attributes:
        agent_encoder (AgentEncoder): encode other agents info
        fov_encoder (FovEncoder): encode self FOV
        fov_encoder_vain (FovEncoder): encode others' FOV
        fov_size (int): fov size
        include_self_attention (bool):
            whether to include self-attention, false in default
        use_k_neighbor (bool): whether to limit neighbors, true in default
        num_neighbors (int): number of neighbors
        without_comm (bool): for ablation study, drop communication feature
    """

    def __init__(
        self,
        agent_encoder: AgentEncoder,
        fov_encoder: FOVEncoder,
        fov_encoder_vain: FOVEncoder,
        include_self_attention: bool = True,
        use_k_neighbor: bool = False,
        num_neighbors: int = 3,
        without_comm: bool = False,
        **kwargs,
    ) -> None:
        if not without_comm:
            dim = 8 + agent_encoder.dim_message + fov_encoder.dim_output
        else:
            dim = 8 + fov_encoder.dim_output

        super().__init__(dim=dim)

        device = get_device_name()

        self.agent_encoder = agent_encoder.to(device)
        self.fov_encoder = fov_encoder.to(device)
        self.fov_encoder_vain = fov_encoder_vain.to(device)
        self.fov_size: int = self.fov_encoder_vain.fov_size
        self.include_self_attention: bool = include_self_attention

        self.use_k_neighbor: bool = use_k_neighbor
        self.num_neighbors: int = num_neighbors
        self.without_comm: bool = without_comm

        # change filanames
        self.fov_encoder.succ_net_name = "_fov_encoder_self.pt"
        self.fov_encoder.succ_hypra_name = "_fov_encoder_self.pkl"
        self.fov_encoder_vain.succ_net_name = "_fov_encoder_vain.pt"
        self.fov_encoder_vain.succ_hypra_name = "_fov_encoder_vain.pkl"

    def __repr__(self) -> str:
        return (
            f"(self_info ({8}), "
            "encoded_fov ({self.fov_encoder.dim_output}), "
            "encoded_comm ({self.agent_encoder.dim_message}))"
        )

    def get_fov_size(self) -> int:
        return self.fov_size

    def get_map_size(self) -> int:
        return self.fov_encoder.map_size

    def get_trainees(self) -> list[Net]:
        """return models to be trained"""
        return [self.agent_encoder, self.fov_encoder, self.fov_encoder_vain]

    def __getstate__(self):
        """called when pickling"""
        state = self.__dict__.copy()
        del state["agent_encoder"]
        del state["fov_encoder"]
        del state["fov_encoder_vain"]
        if "occupancy_map" in state.keys():
            del state["occupancy_map"]
        if "arr_cost_to_go" in state.keys():
            del state["arr_cost_to_go"]
        return state

    def __setstate__(self, state):
        """called when un-pickling"""
        self.__dict__.update(state)

    def save(self, basename: str):
        """save with neural networks"""
        with open(basename + self.succ_format_name, "wb") as fb:
            pickle.dump(self, fb)
        self.agent_encoder.save(basename)
        self.fov_encoder_vain.save(basename)
        self.fov_encoder.save(basename)

    def reconstruct_trainees(self, basename: str):
        """reconstruct neural networks"""
        self.agent_encoder = reconstruct_agent_encoder(basename)
        self.fov_encoder_vain = reconstruct_fov_encoder(
            basename, "_fov_encoder_vain.pkl", "_fov_encoder_vain.pt",
        )
        self.fov_encoder = reconstruct_fov_encoder(
            basename, "_fov_encoder_self.pkl", "_fov_encoder_self.pt",
        )

    def set_instance(self, ins: Instance) -> None:
        """set occupancy map and cost_to_go map"""
        self.occupancy_map = ins.objs.draw_2d(self.fov_encoder.map_size)
        self.arr_cost_to_go = [
            get_approx_cost_to_go_matrix_2d(
                goal=ins.goals[i],
                occupancy_map=self.occupancy_map,
                rad=ins.rads[i],
            )
            for i in range(ins.num_agents)
        ]

    @staticmethod
    @torch.jit.script
    def get_comm(
        T: int,
        num_agents: int,
        num_neighbors: int,
        arr_others_info: torch.Tensor,
        attentions: torch.Tensor,
        messages: torch.Tensor,
        include_self_attention: bool,
    ) -> torch.Tensor:
        """compute k-neighbors, then compute vain result

        Args:
            T (int): maximum timestep
            num_agents (int): number of agents
            num_neighbors (int): number of neighbors
            arr_others_info (torch.Tensor): other agents info
            attentions (torch.Tensor): attention vectors
            messages (torch.Tensor): message vectors
            include_self_attention (:obj:`bool`, optional):
                whether to include self-attention

        Returns:
            torch.Tensor: implementation of Eq. 1 in the paper
        """

        # get k-neighbors indexes
        neighbors_idx = torch.argsort(
            arr_others_info.reshape(T, num_agents, num_agents, -1)[:, :, :, 2]
        ).reshape(T, num_agents, num_agents, 1)
        # obtain k-neighbors' attention
        attentions_sorted = torch.gather(
            attentions.reshape(T, num_agents, num_agents, -1),
            2,
            neighbors_idx.expand(
                T, num_agents, num_agents, attentions.shape[-1]
            ),
        )[:, :, : num_neighbors + 1]
        # obtain k-neighbors' message
        messages_sorted = torch.gather(
            messages.reshape(T, num_agents, num_agents, -1),
            2,
            neighbors_idx.expand(
                T, num_agents, num_agents, messages.shape[-1]
            ),
        )[:, :, 1 : num_neighbors + 1]
        # compute weight
        diff = attentions_sorted - attentions_sorted[:, :, 0].reshape(
            T, num_agents, 1, -1
        )
        pow2 = torch.pow(diff, 2)
        similarities = -torch.sum(pow2, dim=-1)
        if not include_self_attention:
            weight = torch.nn.functional.softmax(
                similarities[:, :, 1:], dim=-1
            )
        else:
            weight = torch.nn.functional.softmax(similarities, dim=-1)[
                :, :, 1:
            ]
        # summation
        return torch.sum(
            messages_sorted * weight.reshape(T, num_agents, num_neighbors, 1),
            dim=2,
        )

    def get_feature_one_step(
        self,
        ins: Instance,
        arr_current_pos: list[np.ndarray],
        arr_prev_pos: list[np.ndarray],
    ) -> torch.Tensor:
        """get feature for one-timestep, used in inference phase

        Args:
            ins (Instance): instance
            arr_current_pos (list[np.ndarray]): current positions
            arr_prev_pos (list[np.ndarray]): previous positions

        Returns:
            torch.Tensor: agent * (self-info, fov_feature, comm_feature)
        """
        num_agents = ins.num_agents

        # compute fov
        arr_fov = [
            get_fov2d_occupancy_cost(
                current_pos=arr_current_pos[j],
                occupancy_map=self.occupancy_map,
                cost_to_go=self.arr_cost_to_go[j],
                fov_size=self.fov_size,
                flatten=True,
            )
            for j in range(num_agents)
        ]
        input_tensor_fov = torch.tensor(arr_fov).float()

        # compute encoded_fov
        encoded_fov_vain = self.fov_encoder_vain(input_tensor_fov)
        encoded_fov = self.fov_encoder(input_tensor_fov)

        # comptue others info
        arr_others_info = torch.tensor(
            get_arr_others_info(
                arr_current_locs=np.array(arr_current_pos).reshape(
                    1, num_agents, -1
                ),
                goals=np.array(ins.goals),
                arr_prev_locs=np.array(arr_prev_pos).reshape(
                    1, num_agents, -1
                ),
                rads=np.array(ins.rads),
                max_speeds=np.array(ins.max_speeds),
            ).reshape(num_agents * num_agents, -1)
        ).float()
        input_tensor_vain = torch.cat(
            [arr_others_info, torch.cat([encoded_fov_vain] * num_agents)],
            dim=1,
        )

        # vain
        messages, attentions = self.agent_encoder(input_tensor_vain)
        num_neighbors = num_agents - 1
        if self.use_k_neighbor:
            num_neighbors = min(self.num_neighbors, num_neighbors)
        encoded_comm = Format2D_CTRM_Input.get_comm(
            T=1,
            num_agents=num_agents,
            num_neighbors=num_neighbors,
            arr_others_info=arr_others_info.reshape(1, *arr_others_info.shape),
            attentions=attentions.reshape(1, *attentions.shape),
            messages=messages.reshape(1, *messages.shape),
            include_self_attention=self.include_self_attention,
        )
        encoded_comm = encoded_comm.reshape(*encoded_comm.shape[1:])

        # compute self info
        self_info = get_self_info(
            num_agents, arr_others_info.reshape(1, *arr_others_info.shape)
        ).reshape(num_agents, -1)

        if "without_comm" not in dir(self) or not self.without_comm:
            return torch.cat([self_info, encoded_fov, encoded_comm], dim=1)
        else:
            return torch.cat([self_info, encoded_fov], dim=1)

    def get_feature(
        self,
        ins: Instance,
        res: Result,
        input_tensor_fov: torch.Tensor,
        arr_others_info: torch.Tensor,
        arr_self_info: torch.Tensor,
    ) -> torch.Tensor:
        """get feature from one instance, used in training phase

        Args:
            ins (Instance): instance
            res (Result): result
            input_tensor_fov (torch.Tensor): time * agent * fov
            arr_others_info (torch.Tensor): time * (self -> other) * feature
            arr_self_info (torch.Tensor): time * agent * feature

        Returns:
            torch.Tensor:
                timestep * agent * (self-info, fov_feature, comm_feature)
        """
        num_agents = ins.num_agents
        T = arr_self_info.shape[0]

        encoded_fov_vain = self.fov_encoder_vain(input_tensor_fov)
        encoded_fov = self.fov_encoder(input_tensor_fov)

        # agent encoder
        input_tensor_vain = torch.cat(
            [arr_others_info, encoded_fov_vain.repeat(1, num_agents, 1)], 2
        )
        messages, attentions = self.agent_encoder(input_tensor_vain)

        # obtain communication
        num_neighbors = num_agents - 1
        if self.use_k_neighbor:
            num_neighbors = min(self.num_neighbors, num_neighbors)

        encoded_comm = Format2D_CTRM_Input.get_comm(
            T=T,
            num_agents=num_agents,
            num_neighbors=num_neighbors,
            arr_others_info=arr_others_info,
            attentions=attentions,
            messages=messages,
            include_self_attention=self.include_self_attention,
        )

        # create feature
        if "without_comm" not in dir(self) or not self.without_comm:
            x = torch.cat([arr_self_info, encoded_fov, encoded_comm], dim=-1)
        else:
            x = torch.cat([arr_self_info, encoded_fov], dim=-1)
        return x.reshape(-1, x.shape[-1])

    def __call__(self, **kwargs) -> torch.Tensor:
        """do not use this func, too slow"""
        pass
