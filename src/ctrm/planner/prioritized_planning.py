"""implementation of a standard prioritized planning
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX

Ref:
- Silver, D. (2005).
  Cooperative Pathfinding.
  Aiide, 1, 117-122.

- Erdmann, M., & Lozano-Perez, T. (1987).
  On multiple moving objects.
  Algorithmica, 2(1), 477-521.
"""

from __future__ import annotations

import heapq

import numpy as np

from ..environment import Instance
from ..roadmap import TimedNode, TimedRoadmap
from .planner import Planner


class PrioritizedPlanning(Planner):
    def __init__(
        self,
        ins: Instance,
        trms: list[TimedRoadmap],
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(ins, trms, verbose, **kwargs)
        self.verbose: int = verbose

    def get_name(self):
        return "PrioritizedPlanning"

    def _solve(self) -> None:
        T = len(self.trms[0].V) - 1  # makespan
        required_timestep = 1

        for agent in range(self.ins.num_agents):
            self.info(f"agent-{agent} starts planning")
            goal_pos = self.ins.goals[agent]
            max_speed = self.ins.max_speeds[agent]
            rad = self.ins.rads[agent]
            trm = self.trms[agent]

            # define search details

            def get_f_value(v: TimedNode) -> float:
                # Note: the value is scaled for the time axis
                return v.t + np.linalg.norm(goal_pos - v.pos) / max_speed

            def check_fin(v: TimedNode) -> bool:
                # the last vertex is goal
                return v.t >= required_timestep and v == trm.V[v.t][-1]

            def insert(v: TimedNode, OPEN: list[list]) -> None:
                # tie-break: f -> g -> random
                heapq.heappush(
                    OPEN, [get_f_value(v), v.t, np.random.rand(), v]
                )

            def valid_successor(v_from: TimedNode, v_to: TimedNode) -> bool:
                return not any(
                    [
                        self.collide_dynamic_agents(
                            v_from.pos,
                            v_to.pos,
                            rad,
                            self.solution[i][v_from.t].pos,
                            self.solution[i][v_to.t].pos,
                            self.ins.rads[i],
                        )
                        for i in range(agent)
                    ]
                )

            # perform space-time A*
            path = self.get_single_agent_path(
                agent, check_fin, insert, valid_successor
            )

            if path is None:  # failed to solve
                self.solution.clear()
                self.info(f"agent-{agent} failed to find paths")
                return

            # update required_timestep (for collision check)
            required_timestep = max(len(path) - 1, required_timestep)

            # format new path, extending by goals
            path += [trm.V[t][-1] for t in range(len(path), T + 1)]

            # update solution
            self.solution.append(path)

        self.solved = True
