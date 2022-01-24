"""NN-models
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

import pickle
from typing import Optional

import torch

from ...utils import get_device_name
from .agent_encoder import AgentEncoder
from .cvae import CTRMNet
from .fov_encoder import FOVEncoder
from .model import Model
from .net import Net


def reconstruct_net(
    basename: str,
    cls: Net,
    succ_hypra_name: Optional[str] = None,
    succ_net_name: Optional[str] = None,
) -> Net:
    """reconstruct neural network

    Args:
        basename (str): e.g., xxxx/best
        cls (Net): class of Net
        succ_hypra_name (:obj:`str`, optional): hyper parameters
        succ_net_name (:obj:`str`, optional): neural network parameters

    Returns:
        Net: neural network
    """
    hypra_file = basename + (
        cls.succ_hypra_name if succ_hypra_name is None else succ_hypra_name
    )
    with open(hypra_file, "rb") as f:
        hypra = pickle.load(f)
        cls_name = hypra[cls.field_cls_name]
        del hypra[cls.field_cls_name]

    net = globals()[cls_name](**hypra)
    device = get_device_name()

    model_file = basename + (
        cls.succ_net_name if succ_net_name is None else succ_net_name
    )
    net.load_state_dict(
        torch.load(open(model_file, "rb"), map_location=torch.device(device),)
    )
    net.eval()
    return net


def reconstruct_model(basename: str) -> Model:
    """reconstruct model"""
    return reconstruct_net(basename, Model)  # type: ignore


def reconstruct_agent_encoder(basename: str) -> AgentEncoder:
    """reconstruct agent encoder"""
    return reconstruct_net(basename, AgentEncoder)  # type: ignore


def reconstruct_fov_encoder(
    basename: str,
    succ_hypra_name: Optional[str] = None,
    succ_net_name: Optional[str] = None,
) -> FOVEncoder:
    """reconstruct fov encoder"""
    return reconstruct_net(  # type: ignore
        basename, FOVEncoder, succ_hypra_name, succ_net_name  # type: ignore
    )


__all__ = [
    "Net",
    "Model",
    "CTRMNet",
    "reconstruct_model",
    "reconstruct_encoder",
    "reconstruct_agent_encoder",
    "AgentEncoder",
    "FOVEncoder",
]
