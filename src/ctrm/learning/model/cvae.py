"""implementation of F_CTRM
Author: Keisuke Okumura / Ryo Yonetani
Affiliation: TokyoTech & OSX / OSX
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from .model import Model


@dataclass(eq=False, repr=False)
class CTRMNet(Model):
    """CVAE to construct CTRMs"""

    dim_input: int
    dim_output: int
    dim_indicators: int = 0  # set automatically in train.py

    # hyper parameters
    dim_hidden: int = 32
    dim_latent: int = 64
    temp: float = 2.0
    num_mid_layers_encoder: int = 1
    num_mid_layers_decoder: int = 1
    kl_weight: float = 0.1  # weighting KL divergence

    def __post_init__(self) -> None:
        super().__init__()

        def generate_mlp(
            dim_input: int, dim_output: int, num_mid_layers: int = 1,
        ) -> nn.modules.container.Sequential:
            return nn.Sequential(
                nn.Linear(dim_input, self.dim_hidden),
                nn.BatchNorm1d(self.dim_hidden),
                nn.ReLU(),
                *(
                    [
                        nn.Linear(self.dim_hidden, self.dim_hidden),
                        nn.BatchNorm1d(self.dim_hidden),
                        nn.ReLU(),
                    ]
                    * num_mid_layers
                ),
                nn.Linear(self.dim_hidden, dim_output),
            )

        def generate_encoder(
            dim_input: int,
        ) -> nn.modules.container.Sequential:
            mlp = generate_mlp(
                dim_input, self.dim_latent, self.num_mid_layers_encoder,
            )
            mlp.add_module("log_softmax", nn.LogSoftmax(dim=-1))
            return mlp

        self.encoder_input = generate_encoder(
            self.dim_input + self.dim_indicators
        )
        self.encoder_output = generate_encoder(
            self.dim_input + self.dim_output
        )
        self.decoder = generate_mlp(
            self.dim_latent + self.dim_input + self.dim_indicators,
            self.dim_output - self.dim_indicators,
            self.num_mid_layers_decoder,
        )
        self.indicator = generate_mlp(
            self.dim_input, self.dim_indicators, self.num_mid_layers_decoder,
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """used in training phase"""
        # predict next location
        assert self.dim_indicators > 0

        # indicator is included in y
        ind = y[:, -self.dim_indicators :].reshape(-1, self.dim_indicators)

        # encode
        augmented_x = torch.cat((x, ind), -1)
        log_prob_x = self.encoder_input(augmented_x)
        log_prob_y = self.encoder_output(torch.cat([x, y], dim=1))
        dist_y = RelaxedOneHotCategorical(
            self.temp, probs=torch.exp(log_prob_y)
        )

        # sampling from the latent space
        latent_y = dist_y.rsample()

        # decode
        y_pred = self.decoder(torch.cat([latent_y, augmented_x], dim=1))

        # indicator prediction
        ind_pred = self.indicator(x)

        # all values are for computing loss
        return y_pred, log_prob_x, log_prob_y, ind_pred

    def predict_with_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """used in training phase"""
        y_pred, log_prob_x, log_prob_y, ind_pred = self.forward(x, y)
        loss_details = self.loss_fn(y, y_pred, log_prob_x, log_prob_y, w)
        loss = reduce(add, loss_details.values())

        # indicator
        ind_pred = nn.LogSoftmax(dim=-1)(ind_pred)
        ind_loss = nn.NLLLoss()(ind_pred, torch.where(y[:, 3:])[1])
        loss = loss + ind_loss * 1e-3
        return y_pred, loss, loss_details

    def sample(self, x: torch.Tensor, ind: torch.Tensor,) -> torch.Tensor:
        """sampling function, used in inference phase"""
        x = torch.cat((x, ind), -1)
        with torch.no_grad():
            log_prob_x = self.encoder_input(x)
            dist_x = RelaxedOneHotCategorical(
                self.temp, probs=torch.exp(log_prob_x)
            )
            latent_x = dist_x.rsample()
            y = self.decoder(torch.cat([latent_x, x], -1))
        return y

    def loss_fn(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        log_prob_x: torch.Tensor,
        log_prob_y: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """compute loss of the model, used in training phase"""
        if self.dim_indicators > 0:
            # indicator is included in y, remove this
            y = y[:, : -self.dim_indicators]

        if weight is None:
            recon_loss = nn.MSELoss()(y_pred, y)
            kl_loss = torch.sum(
                torch.exp(log_prob_x) * (log_prob_x - log_prob_y), dim=-1
            ).mean()
        else:
            weight = weight.reshape(-1)
            recon_loss = (torch.sum((y_pred - y) ** 2, dim=-1) * weight).mean()

            kl_loss = (
                torch.sum(
                    torch.exp(log_prob_x) * (log_prob_x - log_prob_y), dim=-1
                )
                * weight
            ).mean() * self.kl_weight

        return {
            "recon": recon_loss,
            "kl": kl_loss,
        }
