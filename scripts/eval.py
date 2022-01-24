"""evaluate roadmap construction and successive planning
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import asdict
from logging import getLogger

import hydra
import numpy as np
import torch
import tqdm
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import DictConfig

from ctrm.utils import set_device_name, set_global_seeds, stop_watch
from ctrm.viz import simple_plot_2d

logger = getLogger(__name__)


@stop_watch("eval")
@torch.no_grad()
@hydra.main(config_path="conf/eval", config_name="config")
def main(conf: DictConfig) -> None:
    logdir = os.getcwd()
    logger.info(f"result will be saved in {logdir}")

    # set seed
    set_global_seeds(conf.seed)

    # get dataset length
    len_dataset = len(glob.glob(os.path.join(conf.insdir, "*_ins.pkl")))
    num_ins = (
        len_dataset
        if conf.max_eval_num <= 0
        else min(len_dataset, conf.max_eval_num)
    )
    logger.info(f"use {num_ins} instances from {conf.insdir}")
    logger.info(f"planner={conf.planner._target_}")

    # get instance
    def process(i: int):
        set_device_name("cpu")

        # obtain instance
        with open(os.path.join(conf.insdir, f"{i:08d}_ins.pkl"), "rb") as fb:
            ins = pickle.load(fb)

        # generate roadmaps
        @stop_watch(verbose=False)
        def roadmap_gen():
            return instantiate(conf.roadmap, ins)

        trms, elapsed_roadmap_gen = roadmap_gen()

        # record the number of samples
        roadmap_gen_name = conf.roadmap._target_.split(".")[-1]
        if roadmap_gen_name in [
            "get_timed_roadmaps_random_common",
            "get_timed_roadmaps_grid_common",
            "get_timed_roadamaps_SPARS_2d_common",
        ]:
            # without starts/goals
            sample_nums = len(trms[0].V[1]) - 1
        elif roadmap_gen_name in ["get_timed_roadmaps_random_rect"]:
            sample_nums = sum([len(trm.V[1]) - 1 for trm in trms])
        else:
            sample_nums = 0
            for trm in trms:
                for t in range(1, len(trm.V)):
                    sample_nums += len(trm.V[t]) - 1  # without starts/goals

        info_roadmap_gen = {
            "sample_nums": sample_nums,
            "elapsed_roadmap_gen": elapsed_roadmap_gen,
            "cnt_roadmap_static_collide": ins.objs.cnt_static_collide,
            "cnt_roadmap_continuous_collide": ins.objs.cnt_continuous_collide,
            "elapsed_roadmap_static_collide": ins.objs.time_static_collide,
            "elapsed_roadmap_continuous_collide": ins.objs.time_continuous_collide,  # noqa: E501
        }

        # create planner
        planner = instantiate(conf.planner, ins, trms)

        # get result
        res = planner.solve()
        result = asdict(res)
        for key, val in info_roadmap_gen.items():
            result[key] = val
        log = (ins, res, info_roadmap_gen)

        # plot result
        if conf.plot_2d_data:
            simple_plot_2d(
                ins, res.paths, os.path.join(logdir, f"viz_{i:08d}.jpg")
            )
        return result, log

    results = []
    log_data = []
    if "n_jobs" in conf.keys():
        data = Parallel(n_jobs=conf.n_jobs, verbose=1)(
            [delayed(process)(i) for i in range(num_ins)]
        )
        for d in data:
            results.append(d[0])
            log_data.append(d[1])
    else:
        for i in tqdm.tqdm(
            range(num_ins),
            desc=f"planning {num_ins} instances",
            disable=conf.progress_bar_disable,
        ):
            res, log = process(i)
            results.append(res)
            log_data.append(log)

    # save data
    with open(os.path.join(logdir, "eval.pkl"), "wb") as fb:
        pickle.dump(log_data, fb)

    # simple stats
    stats_str = ""

    # success rate
    num_solved = [res["solved"] for res in results].count(True)
    stats_str += format("success rate", ">35s")
    stats_str += f":  {num_solved:04d}/{num_ins:04d}="
    stats_str += f"{num_solved/num_ins:0.4f}\n"

    # other metric
    if num_solved > 0:
        for key in results[0].keys():
            if key in ["solved", "paths", "name_planner"]:
                continue
            arr = [res[key] for res in results if res["solved"]]
            stats_str += f"{key:>35s}:  "
            stats_str += (
                f"mean={np.mean(arr):>8.4f} ± {np.std(arr, ddof=1):>8.4f}  "
            )
            stats_str += f"median={np.median(arr):>8.4f}  "
            stats_str += f"max={np.max(arr):>8.4f}  "
            stats_str += f"min={np.min(arr):>8.4f}\n"

    logger.info("stats:\n" + stats_str)

    # save
    with open(os.path.join(logdir, "stats.txt"), "w") as f:
        f.write(stats_str)


if __name__ == "__main__":
    main()
