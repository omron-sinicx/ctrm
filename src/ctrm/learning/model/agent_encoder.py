"""implementation of f_comm
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import Model


@dataclass(eq=False, repr=False)
class AgentEncoder(Model):
    """agent encoder, used for VAIN"""

    dim_input: int
    dim_message: int = 32
    dim_attention: int = 10
    dim_hidden: int = 32
    num_mid_layers: int = 1
    use_sigmoid: bool = False
    use_batch_norm: bool = False

    succ_net_name: str = "_agent_encoder.pt"
    succ_hypra_name: str = "_agent_encoder.pkl"

    def __post_init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(self.dim_input, self.dim_hidden),
            nn.ReLU(),
            *(
                (
                    [
                        nn.Linear(self.dim_hidden, self.dim_hidden),
                        nn.BatchNorm1d(self.dim_hidden),
                        nn.ReLU(),
                    ]
                    * self.num_mid_layers
                )
                if self.use_batch_norm
                else (
                    [nn.Linear(self.dim_hidden, self.dim_hidden), nn.ReLU(),]
                    * self.num_mid_layers
                )
            ),
            nn.Linear(self.dim_hidden, self.dim_message + self.dim_attention),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """message, attention"""
        y = self.mlp(x)
        message, attention = torch.split(
            y, [self.dim_message, self.dim_attention], dim=-1
        )

        if self.use_sigmoid:
            message = nn.Sigmoid()(message)

        return message, attention
