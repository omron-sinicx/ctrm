"""implementation of Conflict-based Search(CBS)
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX

Ref:
- Sharon, G., Stern, R., Felner, A., & Sturtevant, N. R. (2015).
  Conflict-based search for optimal multi-agent pathfinding.
  Artificial Intelligence, 219, 40-66.
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..environment import Instance
from ..roadmap import TimedNode, TimedRoadmap
from .planner import Planner


@dataclass(frozen=True)
class VertexConstraint:
    """prohibit agent to use v"""

    agent: int
    v: TimedNode


@dataclass(frozen=True)
class EdgeConstraint:
    """prohibit agent to move from v_from to v_to"""

    agent: int
    v_from: TimedNode
    v_to: TimedNode


@dataclass
class HighLevelNode:
    paths: list[list[TimedNode]]  # list of **timed** paths
    constraints: list[Union[VertexConstraint, EdgeConstraint]]
    quality: float  # sum-of-costs, makespan, etc
    num_conflicts: int  # number of conflicts
    # prioritize nodes with small collisions
    prioritize_collision: bool = False

    def __lt__(self, other: HighLevelNode):
        """for priority queue"""
        if not self.prioritize_collision and self.quality != other.quality:
            return self.quality < other.quality
        if not self.num_conflicts != other.num_conflicts:
            return self.num_conflicts < other.num_conflicts
        return len(self.constraints) > len(other.constraints)


class CBS(Planner):
    def __init__(
        self,
        ins: Instance,
        trms: list[TimedRoadmap],
        prioritize_collision: bool = False,
        **kwargs,
    ):
        super().__init__(ins, trms, **kwargs)
        self.prioritize_collision = prioritize_collision

    def get_name(self) -> str:
        return "CBS"

    def _solve(self) -> None:
        """high-level search for CBS"""
        OPEN: list[HighLevelNode] = []

        # get initial node
        init_node = self.get_initial_node()
        if init_node is None:  # failure
            self.info("CBS cannot solve this instance")
            return
        heapq.heappush(OPEN, init_node)

        while len(OPEN) > 0:
            # pop
            node: HighLevelNode = heapq.heappop(OPEN)

            self.info(
                f"expand node, quality={node.quality}, "
                f"num_conflicts={node.num_conflicts}, "
                f"num_constraints={len(node.constraints)}",
            )

            # check conflicts
            constraints = self.get_constraints(node.paths)
            if len(constraints) == 0:  # solved
                self.solution = node.paths
                self.solved = True
                return

            # create new nodes
            for c in constraints:
                new_node = HighLevelNode(
                    copy.deepcopy(node.paths),
                    node.constraints + [c],
                    node.quality,
                    node.num_conflicts,
                    prioritize_collision=self.prioritize_collision,
                )
                if isinstance(c, VertexConstraint):
                    self.info(
                        f"  add new child, agent-{c.agent}"
                        f" at t={c.v.t}, loc={c.v.pos}"
                    )
                elif isinstance(c, EdgeConstraint):
                    self.info(
                        f"  add new child, agent-{c.agent}"
                        f" at t={c.v_from.t} -> t={c.v_to.t}"
                        f" loc={c.v_from.pos} -> {c.v_to.pos}"
                    )

                if self.invoke(new_node):
                    heapq.heappush(OPEN, new_node)
                    self.info("    inserted")

    def invoke(self, node: HighLevelNode) -> bool:
        """setup one high-level search node

        Args:
            node (HighLevelNode): search node

        Retruns:
            bool: false -> invalid
        """
        T = len(self.trms[0].V) - 1  # makespan
        agent = node.constraints[-1].agent
        path = self.lowlevel_serach(agent, node.constraints)
        if path is None:
            return False
        # format, extending by goals
        path += [self.trms[agent].V[t][-1] for t in range(len(path), T + 1)]
        # update node
        node.paths[agent] = path
        node.quality = self.get_quality(node.paths)
        node.num_conflicts = self.get_num_conflicts(node.paths)
        return True

    def get_quality(self, paths: list[list[TimedNode]]) -> float:
        """compute solution quality

        Args:
            paths (list[list[TimedNode]]): paths

        Returns:
            float: sum-of-costs
        """
        return self.get_sum_of_costs(paths)

    def get_constraints(
        self, paths: list[list[TimedNode]]
    ) -> list[Union[VertexConstraint, EdgeConstraint]]:
        """find constraints, otherwise return empty list

        Args:
            paths (list[list[TimedNode]]): paths

        Returns:
            list[Union[VertexConstraint, EdgeConstraint]]: constraints
        """
        T = paths[0][-1].t  # makespan
        for t in range(T + 1):
            for i in range(self.ins.num_agents):
                rad_i = self.ins.rads[i]
                v_i_t = paths[i][t]
                for j in range(i + 1, self.ins.num_agents):
                    rad_j = self.ins.rads[j]
                    v_j_t = paths[j][t]
                    # check vertex conflicts
                    if self.collide_static_agents(
                        v_i_t.pos, rad_i, v_j_t.pos, rad_j
                    ):
                        return [
                            VertexConstraint(i, v_i_t),
                            VertexConstraint(j, v_j_t),
                        ]
                    if t == 0:
                        continue
                    v_i_t_prev = paths[i][t - 1]
                    v_j_t_prev = paths[j][t - 1]
                    if self.collide_dynamic_agents(
                        v_i_t_prev.pos,
                        v_i_t.pos,
                        rad_i,
                        v_j_t_prev.pos,
                        v_j_t.pos,
                        rad_j,
                    ):
                        return [
                            EdgeConstraint(i, v_i_t_prev, v_i_t),
                            EdgeConstraint(j, v_j_t_prev, v_j_t),
                        ]
        return []

    def get_initial_node(self) -> Optional[HighLevelNode]:
        """setup initial node, return None when failing to initialize

        Returns:
            Optional[HighLevelNode]: initial node, otherwise None
        """
        T = len(self.trms[0].V) - 1  # makespan
        paths = []
        for i in range(self.ins.num_agents):
            path = self.lowlevel_serach(i)
            if path is None:
                return None
            # format, extending by goals
            path += [self.trms[i].V[t][-1] for t in range(len(path), T + 1)]
            paths.append(path)
        return HighLevelNode(
            paths,
            [],
            self.get_quality(paths),
            self.get_num_conflicts(paths),
            prioritize_collision=self.prioritize_collision,
        )

    def get_num_conflicts(self, paths: list[list[TimedNode]]) -> int:
        """compute number of conflicts in given paths

        Args:
            paths (list[list[TimedNode]])): paths

        Returns:
            int: number of conflicts
        """
        # avoid checking edge conflicts, it probably takes time
        cnt = 0
        T = paths[0][-1].t  # makespan
        for t in range(T + 1):
            for i in range(self.ins.num_agents):
                rad_i = self.ins.rads[i]
                v_i_t = paths[i][t]
                for j in range(i + 1, self.ins.num_agents):
                    rad_j = self.ins.rads[j]
                    v_j_t = paths[j][t]
                    # check vertex conflicts
                    if self.collide_static_agents(
                        v_i_t.pos, rad_i, v_j_t.pos, rad_j
                    ):
                        cnt += 1
        return cnt

    def lowlevel_serach(
        self,
        agent: int,
        constraints: list[Union[VertexConstraint, EdgeConstraint]] = [],
    ) -> Optional[list[TimedNode]]:
        """low-level search for CBS

        Args:
            agent (int): target agent
            constraints (:obj:`list[Union[VertexConstraint, EdgeConstraint]]`,
                         optional): search constraints

        Returns:
            Optional[list[TimedNode]]: path or None (failure)
        """
        goal_pos = self.ins.goals[agent]
        max_speed = self.ins.max_speeds[agent]
        required_timestep = 1
        filtered_constraints = []
        for c in constraints:
            if c.agent == agent:
                filtered_constraints.append(c)
                if isinstance(c, VertexConstraint):
                    required_timestep = max(c.v.t, required_timestep)
                elif isinstance(c, EdgeConstraint):
                    required_timestep = max(c.v_to.t, required_timestep)

        def get_f_value(v: TimedNode) -> float:
            return v.t + np.linalg.norm(goal_pos - v.pos) / max_speed

        def check_fin(v: TimedNode) -> bool:
            return (
                v.t >= required_timestep and v == self.trms[agent].V[v.t][-1]
            )

        def insert(v: TimedNode, OPEN: list[list]) -> None:
            heapq.heappush(OPEN, [get_f_value(v), v.t, np.random.rand(), v])

        def valid_successor(v_from: TimedNode, v_to: TimedNode) -> bool:
            for c in filtered_constraints:
                if isinstance(c, VertexConstraint):
                    if c.v == v_to:
                        return False
                elif isinstance(c, EdgeConstraint):
                    if c.v_from == v_from and c.v_to == v_to:
                        return False
            return True

        # return A* search result
        return self.get_single_agent_path(
            agent, check_fin, insert, valid_successor
        )
