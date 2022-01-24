"""implementation of fov encoder, f_{env_self}, f_{env_other}
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import Model


@dataclass(eq=False, repr=False)
class FOVEncoder(Model):
    """encoder for environmental information"""

    dim_output: int = 32
    dim_hidden: int = 32
    fov_size: int = 25
    map_size: int = 64
    num_mid_layers: int = 1
    use_sigmoid: bool = False
    use_batch_norm: bool = False

    succ_net_name: str = "_fov_encoder.pt"
    succ_hypra_name: str = "_fov_encoder.pkl"

    def __post_init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear((self.fov_size ** 2) * 2, self.dim_hidden),
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
            nn.Linear(self.dim_hidden, self.dim_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.use_sigmoid:
            x = nn.Sigmoid()(x)
        return x
