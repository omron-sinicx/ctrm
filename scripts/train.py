"""train models
Author: Keisuke Okumura / Ryo Yonetani
Affiliation: TokyoTech & OSX / OSX
"""

from __future__ import annotations

import os
from functools import reduce
from logging import getLogger

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ctrm.learning import (BatchGenerator, CTRMNet, Dataset, FormatInput,
                           FormatOutput, reconstruct, save)
from ctrm.planner import Result
from ctrm.utils import (get_device_name, set_device_name, set_global_seeds,
                        stop_watch)
from ctrm.viz import simple_plot_2d

logger = getLogger(__name__)
OmegaConf.register_new_resolver("concat", lambda x, y: x + y)


@stop_watch("train")
@hydra.main(config_path="conf/train", config_name="config")
def main(conf: DictConfig) -> None:
    # set seed
    set_global_seeds(conf.seed)

    # recognize environment
    if conf.device is not None:
        set_device_name(conf.device)
    device = get_device_name()

    # input/output format: Callable classes
    format_input: FormatInput = instantiate(conf.format_input)
    format_output: FormatOutput = instantiate(conf.format_output)
    dims = {
        "dim_input": format_input.get_dim(),
        "dim_output": format_output.get_dim(),
        "dim_indicators": format_output.get_dim_indicators(),
    }

    # callable class, working as collate_fn
    batch_generator: BatchGenerator = instantiate(
        conf.batch_generator, format_input, format_output,
    )
    logger.info(f"data : {batch_generator}")

    # setup loader
    loaders = {}

    @stop_watch("load_data")
    def load_data():
        for label in ["train", "val", "test"]:
            dataset = instantiate(
                conf.dataset,
                os.path.join(conf.datadir, label),
                fov_size=format_input.get_fov_size(),
                map_size=format_input.get_map_size(),
                format_output=format_output,
            )
            loaders[label] = instantiate(
                conf.dataloader,
                dataset,
                batch_size=min(conf.dataloader.batch_size, len(dataset)),
                collate_fn=batch_generator,
            )

    load_data()

    # define model
    model: CTRMNet = instantiate(conf.model, **dims).to(device)

    # models requiring training, maybe map encoder
    trainees = [model, *format_input.get_trainees()]
    logger.info("trainees=\n")
    for trainee in trainees:
        logger.info(f"\n{trainee}")

    # optimizer
    opt = instantiate(
        conf.optimizer,
        reduce(
            lambda a, b: a + b,
            [list(trainee.parameters()) for trainee in trainees],
        ),
    )

    # for logging
    logdir = os.getcwd()
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tb"))

    # the best one
    best_loss = np.inf
    best_success_rate: float = 0
    best_sum_of_costs: float = float("inf")

    # define single step
    def run(batch):
        input_tensor, output_tensor, weight_tensor = batch
        #
        y_pred, loss, loss_details = model.predict_with_loss(
            x=input_tensor.to(device),
            y=output_tensor.to(device),
            w=weight_tensor.to(device),
        )
        return y_pred, loss, loss_details

    # start learning
    logger.info(f"num_epochs: {conf.num_epochs}")
    logger.info(
        "data size: "
        + ", ".join(
            [f"{key}={len(loader.dataset)}" for key, loader in loaders.items()]
        )
    )

    for epoch in range(conf.num_epochs):

        # training
        map(lambda x: x.train(), trainees)
        train_loss: float = 0
        train_loss_details: dict[str, float] = {}
        for batch in tqdm(
            loaders["train"],
            desc=format(
                f"{epoch+1:04d}/{conf.num_epochs} training".ljust(18, " ")
            ),
            disable=conf.progress_bar_disable,
        ):
            y_pred, loss, loss_details = run(batch)
            opt.zero_grad()  # initialize grad
            loss.backward()  # back prop
            opt.step()  # update params
            train_loss += loss.item()
            for key, val in loss_details.items():
                if key not in train_loss_details.keys():
                    train_loss_details[key] = 0
                train_loss_details[key] += val.item()
        train_loss /= len(loaders["train"])
        writer.add_scalar("loss/train", train_loss, epoch)
        for key in train_loss_details.keys():
            train_loss_details[key] /= len(loaders["train"])
            writer.add_scalar(
                f"loss_details_train/{key}", train_loss_details[key], epoch
            )

        # validation
        map(lambda x: x.eval(), trainees)
        val_loss: float = 0
        val_loss_details: dict[str, float] = {}
        for batch in tqdm(
            loaders["val"],
            desc=format(
                f"{epoch+1:04d}/{conf.num_epochs} validation".ljust(18, " ")
            ),
            disable=conf.progress_bar_disable,
        ):
            with torch.no_grad():
                y_pred, loss, loss_details = run(batch)
                val_loss += loss.item()
                for key, val in loss_details.items():
                    if key not in val_loss_details.keys():
                        val_loss_details[key] = 0
                    val_loss_details[key] += val.item()
        val_loss /= len(loaders["val"])
        writer.add_scalar("loss/val", val_loss, epoch)
        for key in val_loss_details.keys():
            val_loss_details[key] /= len(loaders["val"])
            writer.add_scalar(
                f"loss_details_val/{key}", val_loss_details[key], epoch
            )

        # check point
        if val_loss < best_loss:
            logger.info(
                f"update best score (epoch={epoch:04d}): "
                f"{best_loss:0.4f} -> {val_loss:0.4f}"
            )
            best_loss = val_loss
            save(f"{logdir}/best", model, format_input, format_output)

        # generate intermediate results
        if (
            conf.intermediate.eval
            and epoch > 0
            and (
                epoch % conf.intermediate.freq == 0
                or epoch == conf.num_epochs - 1
            )
        ):
            dataset = Dataset(conf.intermediate.datadir, preprocessing=False)
            T = len(dataset[0][1].paths[0]) - 1
            imgs_trms: list[np.ndarray] = []
            imgs_res: list[np.ndarray] = []
            arr_res: list[Result] = []
            num_eval = (
                len(dataset)
                if conf.intermediate.num_eval <= 0
                else min(len(dataset), conf.intermediate.num_eval)
            )

            for i in tqdm(
                range(num_eval),
                desc=f"evaluate intermediate states, {num_eval}-data, T={T}",
            ):
                ins = dataset[i][0]
                # generate roadmap
                trms = instantiate(
                    conf.intermediate.roadmap_gen, ins, f"{logdir}/best",
                )
                # plot roadmap
                imgs_trms.append(
                    instantiate(conf.intermediate.roadmap_viz, ins, trms)
                )
                # solve
                planner = instantiate(conf.intermediate.planner, ins, trms)
                res = planner.solve()
                arr_res.append(res)
                imgs_res.append(
                    simple_plot_2d(  # type: ignore
                        ins, res.paths, return_numpy=True
                    )
                )
            # compute metrics
            success_rate = (
                len([res for res in arr_res if res.solved]) / num_eval
            )
            sum_of_costs = np.mean(
                [res.sum_of_costs for res in arr_res if res.solved]
            )
            elapsed_planner = np.mean(
                [res.elapsed_planner for res in arr_res if res.solved]
            )
            elapsed_planner = np.mean(
                [res.elapsed_planner for res in arr_res if res.solved]
            )
            # update checkpoints
            if success_rate > 0 and sum_of_costs < best_sum_of_costs:
                logger.info(
                    "update best score w.r.t. "
                    f"sum_of_costs (epoch={epoch:04d}): "
                    f"{best_sum_of_costs:0.4f} -> {sum_of_costs:0.4f}"
                )
                best_sum_of_costs = float(sum_of_costs)
                save(f"{logdir}/best_soc", model, format_input, format_output)
            if success_rate > 0 and success_rate >= best_success_rate:
                logger.info(
                    "update best score w.r.t. "
                    f"success_rate (epoch={epoch:04d}): "
                    f"{best_success_rate:0.4f} -> {success_rate:0.4f}"
                )
                best_success_rate = success_rate
                save(
                    f"{logdir}/best_success_rate",
                    model,
                    format_input,
                    format_output,
                )

            writer.add_scalar("metric/success_rate", success_rate, epoch)
            writer.add_scalar("metric/sum_of_costs", sum_of_costs, epoch)
            writer.add_scalar("metric/elapsed_planner", elapsed_planner, epoch)
            writer.add_images(
                "images/roadmap",
                torch.tensor(imgs_trms),
                epoch,
                dataformats="NHWC",
            )
            writer.add_images(
                "images/planned_paths",
                torch.tensor(imgs_res),
                epoch,
                dataformats="NHWC",
            )

    # save final state
    logger.info(f"save final state (epoch={epoch:04d}): {val_loss:0.4f}")
    save(f"{logdir}/final", model, format_input, format_output)

    # test
    model, _, _ = reconstruct(f"{logdir}/best")  # type: ignore
    model.to(device)

    test_loss = []
    for batch in tqdm(
        loaders["test"],
        desc=format(f"{epoch+1:04d}/{conf.num_epochs} test").ljust(18, " "),
        disable=conf.progress_bar_disable,
    ):
        with torch.no_grad():
            y_pred, loss, _ = run(batch)
            test_loss.append(loss.item())
    np.save(f"{logdir}/test_loss.npy", test_loss)

    logger.info(f"test loss: mean={np.mean(test_loss)}")
    logger.info(f"saved in {logdir}")
    writer.close()


if __name__ == "__main__":
    main()
