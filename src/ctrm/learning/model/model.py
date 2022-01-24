"""template of NN models
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from .net import Net


class Model(Net):
    """template of model"""

    succ_net_name: str = "_model.pt"
    succ_hypra_name: str = "_model_hypra.pkl"

    def __init__(self):
        super().__init__()
