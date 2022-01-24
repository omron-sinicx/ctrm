import numpy as np

from ctrm.roadmap import TimedRoadmap


def test_timed_roadmap():
    def valid_edge(pos1, pos2):
        return np.linalg.norm(pos1 - pos2) < 0.1

    trm = TimedRoadmap(np.array([0, 0]))
    assert len(trm.V) == 1 and len(trm.V[0]) == 1
    assert len(trm.E) == 1 and len(trm.E[0]) == 1
    assert len(trm.E[0][0]) == 0

    # append t=1
    trm.append_samples([np.array([0, 0]), np.array([0, 1])], 1, valid_edge)
    assert len(trm.V) == 2 and len(trm.V[1]) == 2
    assert len(trm.E) == 2 and len(trm.E[1]) == 2
    assert len(trm.E[0][0]) == 1 and trm.E[0][0][0] == 0
