from __future__ import annotations

import os

from ctrm.environment import Instance
from ctrm.learning import Dataset
from ctrm.planner import Result

datadir = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "sample", "train",
)


def test_Dataset():
    dataset = Dataset(datadir)
    ins, res = dataset[0]

    assert len(dataset) == 10
    assert isinstance(ins, Instance)
    assert isinstance(res, Result)
