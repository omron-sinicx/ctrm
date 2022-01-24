"""template of formatter
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

import pickle
from abc import ABCMeta, abstractmethod

import torch


class Format(metaclass=ABCMeta):
    """template of input/output format"""

    succ_format_name: str = ""

    def __init__(self, dim: int, **kwargs):
        self.dim: int = dim  # dimension

    def get_dim(self) -> int:
        return self.dim

    @abstractmethod
    def __repr__(self) -> str:
        """info of extracted data"""
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor:
        pass

    def save(self, basename: str):
        """save as pickle"""
        with open(basename + self.succ_format_name, "wb") as fb:
            pickle.dump(self, fb)
