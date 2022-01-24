"""batch generator
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import torch

from ..utils import get_device_name
from .formats import FormatInput, FormatOutput


class BatchGenerator(metaclass=ABCMeta):
    """template of batch generator

    used in collate_fn of Data loader

    Attributes:
        format_input (FormatInput): outputs input vector x
        format_output (FormatOutput): outputs target vector y
    """

    def __init__(self, format_input: FormatInput, format_output: FormatOutput):
        self.format_input: FormatInput = format_input
        self.format_output: FormatOutput = format_output

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __call__(
        self, batch_original: list[tuple[Any, ...]],
    ) -> tuple[torch.Tensor, ...]:  # input, output
        pass


class FastBatchGenerator(BatchGenerator):
    """batch generator specialized for CTRMNet to speedup the learning

    Attributes:
        sampling_rate (float): in [0, 1], limits the number of samples
    """

    def __init__(
        self,
        format_input: FormatInput,
        format_output: FormatOutput,
        sampling_rate: float = 1.0,
    ) -> None:
        super().__init__(format_input, format_output)
        self.sampling_rate: float = sampling_rate

    def __repr__(self):
        return f"input={self.format_input}, output={self.format_output}"

    def __call__(
        self, batch_original: list[tuple[Any, ...]],
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            batch_original (list[tuple[Any, ...]]):
                a list of tuples (Instance, Result, FOV, other agents info,
                                  self info, y, weight)
                check also Dataset.__get_item__

        Returns:
            torch.Tensor: input feature (2D)
            torch.Tensor: output feature (2D)
            torch.Tensor: weight (1D)
        """
        X = []
        Y = []
        W = []

        device = get_device_name()
        for k, data in enumerate(batch_original):
            ins, res = data[0], data[1]
            T = int(res.maximum_costs)
            num_samples = int((T) * self.sampling_rate)
            arr_timestep = np.random.choice(
                np.arange(T), num_samples, replace=False
            )
            X.append(
                self.format_input.get_feature(
                    ins=ins,
                    res=res,
                    input_tensor_fov=data[2][arr_timestep].to(device),
                    arr_others_info=data[3][arr_timestep].to(device),
                    arr_self_info=data[4][arr_timestep].to(device),
                )
            )
            Y.append(data[5][arr_timestep].reshape(-1, data[5].shape[-1]))
            W.append(data[6][arr_timestep].reshape(-1, 1))

        return torch.cat(X), torch.cat(Y), torch.cat(W)
