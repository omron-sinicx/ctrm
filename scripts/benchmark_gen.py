"""generate benchmarks
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


@stop_watch("benchmark_gen")
@hydra.main(config_path="conf/benchmark_gen", config_name="config")
def main(conf: DictConfig) -> None:
    datadir = os.getcwd()
    logger.info(f"result will be saved in {datadir}")

    seed_start = conf.seed

    def process(i: int, seed: int):
        # set seed
        set_global_seeds(seed)

        # generate instance
        ins = instantiate(conf.instance)

        # save instance, roadmaps, result
        simple_plot_2d(
            ins, filename=os.path.join(datadir, f"{i:08d}_viz.jpg"),
        )
        with open(os.path.join(datadir, f"{i:08d}_ins.pkl"), "wb") as f:
            pickle.dump(ins, f)

    logger.info(f"generate {conf.data_num} instances")
    if conf.n_jobs in [0, 1]:
        for i in tqdm.tqdm(range(conf.data_num)):
            process(i, i + seed_start)
    else:
        Parallel(n_jobs=conf.n_jobs, verbose=1)(  # parallelization
            [delayed(process)(i, i + seed_start) for i in range(conf.data_num)]
        )
    logger.info("fin")


if __name__ == "__main__":
    main()
