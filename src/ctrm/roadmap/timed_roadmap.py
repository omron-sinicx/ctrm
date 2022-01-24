"""timed roadmaps
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TimedNode:
    """vertex (space-time pair)"""

    t: int  # time
    index: int  # index within the timed roadmap
    pos: np.ndarray  # position

    def __eq__(self, other) -> bool:
        if type(other) is not TimedNode:
            return False
        return (
            self.t == other.t
            and self.index == other.index
            and np.array_equal(self.pos, other.pos)
        )

    def __getstate__(self):
        """called when pickling, reducing volume"""
        state = (self.t, self.index, self.pos)
        return state

    def __setstate__(self, state):
        """called when un-pickling"""
        object.__setattr__(self, "t", state[0])
        object.__setattr__(self, "index", state[1])
        object.__setattr__(self, "pos", state[2])


class TimedRoadmap:
    """timed roadmaps

    Attributes:
        V (list[list[TimedNode]]): vertices [t][i]
        E (list[list[list[int]]]): adj_list [t][i] -> index list of t+1
        dim (int): dimension of the instance
    """

    def __init__(self, loc_start: np.ndarray):
        self.V: list[list[TimedNode]] = [[TimedNode(0, 0, loc_start)]]
        self.E: list[list[list[int]]] = [[[]]]
        self.dim = loc_start.shape[0]

    def __repr__(self) -> str:
        info: str = ""
        for t in range(len(self.V)):
            for i, v in enumerate(self.V[t]):
                info += f"{i}, {self.V[t][i]}, children={self.E[t][i]}\n"
            info += "\n"
        return info

    def append_sample(
        self,
        loc: np.ndarray,
        t: int,
        valid_edge: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
        parents: Optional[list[int]] = None,
        children: Optional[list[int]] = None,
    ) -> None:
        """append one sample to timed roadmap

        Args:
            loc (np.ndarray): location
            t (int): time
            valid_edge (:obj:`Optional[Callable[[np.ndarray,
                                                 np.ndarray], bool]]`,
                        optional): definition of valid move
            parents (:obj:`Optional[list[int]]`, optional): index of parents
            children (:obj:`Optional[list[int]]`, optional): index of children
        """

        assert t > 0, "use __init__ instead"
        self.extend_layer(t)
        new_node = TimedNode(t, len(self.V[t]), loc)
        self.V[t].append(new_node)
        self.E[t].append([])

        # update neighbors
        if parents is None and valid_edge is not None:
            self.add_parents(new_node, valid_edge)
        elif parents is not None:
            for p in parents:
                self.E[t - 1][p].append(new_node.index)

        if children is None and valid_edge is not None:
            self.add_children(new_node, valid_edge)
        elif children is not None:
            for p in children:
                self.E[t][new_node.index].append(p)

    def extend_layer(self, t: int) -> None:
        """extend timed roadmap until a certain timestep

        Args:
            t (int): timestep
        """
        while t > len(self.V) - 1:
            self.V.append([])
            self.E.append([])

    def add_parents(
        self,
        new_node: TimedNode,
        valid_edge: Callable[[np.ndarray, np.ndarray], bool],
    ) -> None:
        """add parents for new vertex

        Args:
            new_node (TimedNode): new vertex
            valid_edge (Callable[[np.ndarray, np.ndarray], bool]):
                definition of valid move
        """
        if new_node.t < 1:
            return
        for i, u in enumerate(self.V[new_node.t - 1]):
            if valid_edge(u.pos, new_node.pos):
                self.E[new_node.t - 1][i].append(new_node.index)

    def add_children(
        self,
        new_node: TimedNode,
        valid_edge: Callable[[np.ndarray, np.ndarray], bool],
    ) -> None:
        """add children for new vertex

        Args:
            new_node (TimedNode): new vertex
            valid_edge (Callable[[np.ndarray, np.ndarray], bool]):
                definition of valid move
        """
        if new_node.t + 1 > len(self.V) - 1:
            return
        for i, u in enumerate(self.V[new_node.t + 1]):
            if valid_edge(new_node.pos, u.pos):
                self.E[new_node.t][new_node.index].append(i)

    def append_samples(
        self,
        locs: list[np.ndarray],
        t: int,
        valid_edge: Callable[[np.ndarray, np.ndarray], bool],
    ) -> None:
        """append a set of samples to timed roadmap

        Args:
            locs (list[np.ndarray]): location
            t (int): time
            valid_edge (Optional[Callable[[np.ndarray, np.ndarray], bool]]):
                definition of valid move
        """
        for loc in locs:
            self.append_sample(loc, t, valid_edge)

    def append_fixed_strucutre(
        self,
        locs: list[np.ndarray],
        T: int,
        valid_edge: Callable[[np.ndarray, np.ndarray], bool],
        adjs: Optional[list[list[int]]] = None,
    ) -> None:
        """append a set of samples until a certain timestep

        Args:
            locs (list[np.ndarray]): locations
            T (int): assumed makespan
            valid_edge (Callable[[np.ndarray, np.ndarray], bool]):
                definition of valid move
            adjs (:obj:`Optional[list[list[int]]]`, optional):
                adjacency array if exists
        """
        assert len(self.V) == 1  # only start loc

        # initial layer
        for i, loc in enumerate(locs):
            self.append_sample(loc, 1, valid_edge)

        # compute adjacency
        if adjs is None:
            adjs = [[i] for i in range(len(locs))]
            for i in range(len(locs)):
                for j in range(i + 1, len(locs)):
                    if valid_edge(locs[i], locs[j]):
                        adjs[i].append(j)
                        adjs[j].append(i)

        # append to body
        self.extend_layer(T)
        for t in range(2, T + 1):
            # add vertices
            for i, loc in enumerate(locs):
                self.V[t].append(TimedNode(t, i, loc))
            # add edges
            self.E[t - 1] = adjs

        self.E[T] = [[] for _ in range(len(locs))]

    def get_parents(self, u: TimedNode) -> list[int]:
        """get parents for a given vertex

        Args:
            u (TimedNode): vertex

        Returns:
            list[int]: indexes of parents

        Note:
            children can be easily obtained

        Todo:
            memorize
        """
        if u.t < 1:
            return []
        return [
            v.index
            for v in self.V[u.t - 1]
            if u.index in self.E[u.t - 1][v.index]
        ]

    def getNodesPosOnestep(self, t: int) -> np.ndarray:
        """get position array for a certain timestep

        Args:
            t (int): timestep

        Returns:
            np.ndarray
        """
        if t >= len(self.V):
            return np.empty((0, self.dim))
        return np.array([u.pos for u in self.V[t]])

    def extend_until(
        self, T: int, valid_edge: Callable[[np.ndarray, np.ndarray], bool]
    ) -> None:
        """extend the current structure until a certain timestep

        Args:
            T (int): assumed makespan
            valid_edge (Callable[[np.ndarray, np.ndarray], bool]):
                definition of valid move
        """
        # compute adjacency
        num_locs = len(self.V[-1])
        adjs = [[i] for i in range(num_locs)]
        for i in range(num_locs):
            for j in range(i + 1, num_locs):
                if valid_edge(self.V[-1][i].pos, self.V[-1][j].pos):
                    adjs[i].append(j)
                    adjs[j].append(i)

        while len(self.V) - 1 < T:
            self.V.append(
                [TimedNode(v.t + 1, v.index, v.pos) for v in self.V[-1]]
            )
            self.E[-1] = copy.deepcopy(adjs)
            self.E.append([[] for _ in range(num_locs)])
