"""template of neural networks
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

import pickle
from dataclasses import asdict, is_dataclass

import torch
import torch.nn as nn


class Net(nn.Module):
    """template of neural network"""

    succ_net_name: str = "_net.pt"
    succ_hypra_name: str = "_hypra.pkl"
    field_cls_name: str = "__name__"

    def __init__(self):
        super().__init__()

    def save(self, basename: str) -> None:
        # save torch params
        torch.save(self.state_dict(), basename + self.succ_net_name)

        # save other params
        if is_dataclass(self):
            with open(basename + self.succ_hypra_name, "wb") as f:
                state = asdict(self)
                state[self.field_cls_name] = type(self).__name__  # class name
                pickle.dump(state, f)
