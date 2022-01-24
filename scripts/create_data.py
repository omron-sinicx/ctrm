"""create MAPP demonstrations
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

import os
import pickle
from logging import getLogger

import hydra
import tqdm
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import DictConfig

from ctrm.utils import set_global_seeds, stop_watch
from ctrm.viz import simple_plot_2d

logger = getLogger(__name__)


@stop_watch("create_data")
@hydra.main(config_path="conf/data_gen", config_name="config")
def main(conf: DictConfig) -> None:
    datadir_parent = os.getcwd()
    logger.info(f"result will be saved in {datadir_parent}")

    # generate data
    seed_start = conf.seed
    for label, num_data in conf.data_nums.items():
        # create directory
        datadir_child = os.path.join(datadir_parent, label)
        os.mkdir(datadir_child)

        def process(i: int, seed: int):
            # set seed
            set_global_seeds(seed)

            while True:
                # generate instance
                ins = instantiate(conf.instance)

                # generate roadmaps
                trms = instantiate(conf.roadmap, ins)

                # create planner
                planner = instantiate(conf.planner, ins, trms)

                # get result
                res = planner.solve()

                # save instance, roadmaps, result
                simple_plot_2d(
                    ins,
                    res.paths,
                    filename=os.path.join(datadir_child, f"{i:08d}_viz.jpg"),
                )
                with open(
                    os.path.join(datadir_child, f"{i:08d}_ins.pkl"), "wb"
                ) as f:
                    pickle.dump(ins, f)
                with open(
                    os.path.join(datadir_child, f"{i:08d}_res.pkl"), "wb"
                ) as f:
                    pickle.dump(res, f)

                # exclude_failure: True -> repeat until success
                if not conf.exclude_failure or res.solved:
                    break

        logger.info(f"generate {label}-data x {num_data}")
        if conf.n_jobs in [0, 1]:
            for i in tqdm.tqdm(range(num_data)):
                process(i, i + seed_start)
        else:
            Parallel(n_jobs=conf.n_jobs, verbose=1)(  # parallelization
                [delayed(process)(i, i + seed_start) for i in range(num_data)]
            )
        seed_start += num_data
    logger.info("fin")


if __name__ == "__main__":
    main()
