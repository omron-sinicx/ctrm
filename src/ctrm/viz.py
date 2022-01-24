"""utilities of visualization
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .environment import Instance, ObstacleBox, ObstacleSphere
from .learning import Dataset
from .planner import get_cost
from .roadmap import TimedNode, TimedRoadmap

COLORS = list(mcolors.TABLEAU_COLORS)


def simple_plot_2d(
    ins: Instance,
    solution: list[list[TimedNode]] = None,
    filename: str = "",
    output_size: int = 2,
    return_numpy: bool = False,
) -> Optional[np.ndarray]:
    """visualize instance and solution

    Args:
        ins (Instance): problem instance
        solution (:obj:`list[list[TimedNode]`, optional): paths
        filename (:obj:`str`, optional):
            len(filename) > 0 -> save the plot as filename
        output_size (:obj:`int`, optional): figure size
        return_numpy (:obj:`bool`, optional):
            true -> return numpy array of the plot

    Returns:
        Optional[np.ndarray]: numpy array of the plot
    """

    fig = plt.figure(figsize=(output_size, output_size))
    ax = fig.add_subplot(111)
    arrow_head = 0.02

    # plot obstacles
    for o in ins.obs:
        if isinstance(o, ObstacleSphere):
            ax.add_patch(plt.Circle(o.pos, o.rad, fc="black"))
        if isinstance(o, ObstacleBox):
            ax.add_patch(
                plt.Rectangle(
                    o.pos - o.size / 2, o.size[0], o.size[1], fc="black"
                )
            )

    # plot solution
    if solution is not None and len(solution) > 0:
        path_len = (
            int(
                max(
                    [
                        get_cost(solution[i], ins.goals[i], ins.goal_rads[i])
                        for i in range(ins.num_agents)
                    ]
                )
            )
            + 1
        )
        for i, path in enumerate(solution):
            color = COLORS[i % len(COLORS)]
            jit = np.random.rand(2) * 0.02 - 0.01
            rad = ins.rads[i]
            s = path[0].pos + jit
            g = path[-1].pos + jit
            for t in range(path_len):
                u = path[t].pos + jit
                alpha = 1 - (0.4 / path_len) * t - 0.55
                ax.add_patch(
                    plt.Circle(u, rad, fc=color, alpha=alpha, ec=color)
                )
                if t == len(path) - 1:
                    continue
                v = path[t + 1].pos + jit
                ax.arrow(
                    u[0],
                    u[1],
                    v[0] - u[0],
                    v[1] - u[1],
                    color=color,
                    head_width=arrow_head,
                    length_includes_head=True,
                )

    # plot start and goal
    for i in range(ins.num_agents):
        color = COLORS[i % len(COLORS)]
        s = ins.starts[i]
        g = ins.goals[i]
        rad = ins.rads[i]
        # start
        if solution is None or len(solution) == 0:
            ax.add_patch(plt.Circle(s, rad, fc=color, alpha=0.45, ec=color))
        ax.text(s[0], s[1], i, size=20)
        ax.scatter([s[0]], [s[1]], marker="o", color=color, s=40)
        # goal
        ax.scatter([g[0]], [g[1]], marker="x", color=color, s=40)

    # set axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if return_numpy:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    if len(filename) > 0:
        plt.savefig(
            filename, pad_inches=0.05, transparent=False, bbox_inches="tight"
        )
    else:
        plt.show()

    return None


def plot_trm_onestep(
    trm: TimedRoadmap,
    ins: Instance,
    filename: str = "",
    t: int = 1,
    output_size: int = 2,
) -> None:
    """visualize one timed roadmap for one timestep

    Args
        trm (TimedRoadmap): one timed roadmap
        ins (Instance): instance
        filename (:obj:`str`, optional):
            len(filename) > 0 -> save the plot as filename
        t (:obj:`int`, optional): timestep
        output_size (:obj:`int`, optional): figure size
    """
    fig = plt.figure(figsize=(output_size, output_size))
    ax = fig.add_subplot(1, 1, 1)

    # plot obstacles
    for o in ins.obs:
        if isinstance(o, ObstacleSphere):
            ax.add_patch(plt.Circle(o.pos, o.rad, fc="black", alpha=0.3))
        if isinstance(o, ObstacleBox):
            ax.add_patch(
                plt.Rectangle(
                    o.pos - o.size / 2, o.size[0], o.size[1], fc="black"
                )
            )

    # plot roadmap
    locs_t0 = np.array([v.pos for v in trm.V[t]])
    X_t0, Y_t0 = locs_t0[:, 0], locs_t0[:, 1]

    locs_t1 = np.array([v.pos for v in trm.V[t + 1]])
    X_t1, Y_t1 = locs_t1[:, 0], locs_t1[:, 1]

    ax.scatter(X_t0, Y_t0, marker=".", s=3)

    for i, neighbors in enumerate(trm.E[t]):
        for j in neighbors:
            ax.plot(
                [X_t0[i], X_t1[j]],
                [Y_t0[i], Y_t1[j]],
                color="black",
                linewidth=0.1,
            )

    # set axis
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if len(filename) > 0:
        plt.savefig(
            filename, pad_inches=0.05, transparent=False, bbox_inches="tight"
        )
    else:
        plt.show()


def plot_trms_all_step(
    ins: Instance,
    trms: list[TimedRoadmap],
    filename: str = "",
    output_size: int = 2,
    return_numpy: bool = False,
) -> Optional[np.ndarray]:
    """visualize all timed roadmaps

    Args:
        ins (Instance): instance
        trms (list[TimedRoadmap): timed roadmaps
        filename (:obj:`str`, optional):
            len(filename) > 0 -> save the plot as filename
        output_size (:obj:`int`, optional): figure size
        return_numpy (:obj:`bool`, optional):
            true -> return numpy array of the plot

    Returns:
        Optional[np.ndarray]: numpy array of the plot
    """

    fig = plt.figure(figsize=(output_size, output_size))
    ax = fig.add_subplot(1, 1, 1)
    T = len(trms[0].V) - 1

    # plot obstacles
    for o in ins.obs:
        if isinstance(o, ObstacleSphere):
            ax.add_patch(plt.Circle(o.pos, o.rad, fc="black", alpha=0.3))
        if isinstance(o, ObstacleBox):
            ax.add_patch(
                plt.Rectangle(
                    o.pos - o.size / 2, o.size[0], o.size[1], fc="black"
                )
            )

    for k, trm in enumerate(trms):
        color = COLORS[k % len(COLORS)]
        for t in range(0, T):
            locs_t0 = np.array([v.pos for v in trm.V[t]])
            X_t0, Y_t0 = locs_t0[:, 0], locs_t0[:, 1]
            locs_t1 = np.array([v.pos for v in trm.V[t + 1]])
            X_t1, Y_t1 = locs_t1[:, 0], locs_t1[:, 1]

            alpha = 1
            ax.scatter(X_t0, Y_t0, marker=".", s=3, color=color, alpha=alpha)
            for i, neighbors in enumerate(trm.E[t]):
                for j in neighbors:
                    ax.plot(
                        [X_t0[i], X_t1[j]],
                        [Y_t0[i], Y_t1[j]],
                        color=color,
                        linewidth=0.3,
                        alpha=alpha,
                    )

    for i in range(ins.num_agents):
        color = COLORS[i % len(COLORS)]
        s = ins.starts[i]
        g = ins.goals[i]
        rad = ins.rads[i]
        # start
        ax.add_patch(plt.Circle(s, rad, fc=color, alpha=0.45, ec=color))
        ax.text(s[0], s[1], i, size=20)
        ax.scatter([s[0]], [s[1]], marker="o", color=color, s=40)
        # goal
        ax.scatter([g[0]], [g[1]], marker="x", color=color, s=40)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # c.f., https://stackoverflow.com/questions/7821518/
    if return_numpy:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    if len(filename) > 0:
        plt.savefig(
            filename, pad_inches=0.05, transparent=False, bbox_inches="tight"
        )
    else:
        plt.show()

    return None


def plot_dataset(
    dataset_dir: str,
    filename: str = "",
    max_num: Optional[int] = None,
    output_size: int = 2,
    **dataset_args,
) -> None:
    """visualize dataset

    Args:
        dataset_dir (str): dataset directory, e.g., xxxx/train/
        filename (:obj:`str`, optional):
            len(filename) > 0 -> save the plot as filename
        max_num (:obj:`Optional[int]`, optional): number of instances to plot
        output_size (:obj:`int`, optional): figure size
        **dataset_args: dataset arguments, see ctrm.learning.Dataset
    """

    dataset = Dataset(dataset_dir, **dataset_args)
    data_num = len(dataset)
    if max_num is not None and max_num > 0:
        data_num = min(max_num, data_num)
    height = int(math.ceil(data_num / 10))

    fig = plt.figure(figsize=(output_size * 10, height * output_size))
    for ins_index in range(data_num):
        ins, res = dataset[ins_index]
        img = simple_plot_2d(ins, res.paths, return_numpy=True)
        ax = fig.add_subplot(height, 10, ins_index + 1)
        ax.imshow(img)
        ax.axis("off")

    fig.subplots_adjust(wspace=0, hspace=0)

    if len(filename) > 0:
        plt.savefig(
            filename, pad_inches=0.05, transparent=False, bbox_inches="tight"
        )
    else:
        plt.show()
