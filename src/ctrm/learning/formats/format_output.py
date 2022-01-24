"""formatting the output feature
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

import pickle
from abc import ABCMeta, abstractmethod

import torch

from ...environment import Instance
from ...planner import Result
from .base import Format
from .utils import get_nextvec_mag


class FormatOutput(Format, metaclass=ABCMeta):
    """output format"""

    succ_format_name: str = "_format_output.Pl"

    @abstractmethod
    def get_feature(
        self,
        ins: Instance,
        res: Result,
        arr_next_info: torch.Tensor,
        arr_direction_sin: torch.Tensor,
        arr_direction_cos: torch.Tensor,
    ) -> torch.Tensor:
        """used with learning process"""
        pass

    @abstractmethod
    def get_dim_indicators(self) -> int:
        """dimension of indicator"""
        return 0


def reconstruct_format_output(basename) -> FormatOutput:
    with open(basename + FormatOutput.succ_format_name, "rb") as fb:
        return pickle.load(fb)


class Format2D_NextVec(FormatOutput):
    """toy example, return goal vector"""

    def __init__(self, **kwargs) -> None:
        super().__init__(dim=2 + 1)

    def __repr__(self) -> str:
        return "(next_pos - current_pos, mag)"

    def __call__(self, **kwargs) -> torch.Tensor:
        return torch.Tensor(get_nextvec_mag(**kwargs)).float()


class Format2D_CTRM_Output(FormatOutput):
    """output feature of CTRMNet

    Attributes:
        dim (int): dimension of output feature
        num_divide (int): how many divide the angles
        use_back (bool):
            prepare specialized dimension for back action
            false in default
    """

    def __init__(
        self, num_divide: int = 2, use_back: bool = False, *args, **kwargs
    ) -> None:
        dim = 3 + 1 + num_divide if use_back else 3 + num_divide
        super().__init__(dim=dim)
        self.use_back: bool = use_back
        self.num_divide: int = num_divide

    def __repr__(self) -> str:
        return f"(next_vec ({3}), indicator ({self.get_dim_indicators()}))"

    def get_dim_indicators(self) -> int:
        return self.num_divide + 1 if self.use_back else self.num_divide

    def get_feature(
        self,
        ins: Instance,
        res: Result,
        arr_next_info: torch.Tensor,
        arr_direction_sin: torch.Tensor,
        arr_direction_cos: torch.Tensor,
    ) -> torch.Tensor:
        """get feature from one demonstration

        Args:
            ins (Instance): instance
            res (Result): result
            arr_next_info (torch.Tensor): time * agent * (goal_vec)
            arr_direction_sin (torch.Tensor): time * agent * sin
            arr_direction_cos (torch.Tensor): time * agent * cos

        Returns:
            torch.Tensor: time * agent * (goal_info, indicator)

        Note:
            Check also Dataset class
        """
        # get one hot vector
        if self.use_back:
            ind = torch.floor(
                (arr_direction_sin + 1) / 2 * self.num_divide + 1
            ).to(torch.long)
            ind[arr_direction_cos <= 0] = 0
            ind = torch.clip(ind, 0, self.num_divide)
            one_hot_vec = torch.nn.functional.one_hot(
                ind, num_classes=self.num_divide + 1
            )
        else:
            ind = torch.floor(
                (arr_direction_sin + 1) / 2 * self.num_divide
            ).to(torch.long)
            ind = torch.clip(ind, 0, self.num_divide - 1)
            one_hot_vec = torch.nn.functional.one_hot(
                ind, num_classes=self.num_divide
            )
        return torch.cat([arr_next_info, one_hot_vec], dim=-1)

    def __call__(self, **kwargs) -> torch.Tensor:
        pass
