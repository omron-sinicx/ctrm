"""result of planning
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ctrm.roadmap import TimedNode


@dataclass(frozen=True)
class Result:
    solved: bool = False  # true -> solved
    paths: list[list[TimedNode]] = field(
        default_factory=list
    )  # solution paths
    name_planner: str = ""  # planner name
    elapsed_planner: float = 0  # runtime of planner
    sum_of_costs: float = 0
    maximum_costs: float = 0  # aka. makespan
    sum_of_travel_dists: float = 0
    maximum_travel_dists: float = 0

    # collision check, planner
    cnt_static_collide: int = 0
    cnt_continuous_collide: int = 0
    elapsed_static_collide: float = 0
    elapsed_continuous_collide: float = 0

    # planning effort
    lowlevel_expanded: int = 0
    lowlevel_explored: int = 0

    def get_dict_wo_paths(self) -> dict[Any, Any]:
        dic = asdict(self)
        del dic["paths"]
        return dic
