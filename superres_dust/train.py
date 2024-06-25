import os
from argparse import ArgumentParser
from pathlib import Path

import wandb
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from metrics import (
    get_extra_metrics,
    get_extra_metrics_on_inputs,
    get_metrics,
    get_metrics_on_inputs,
)
from datasets import SRDDataModule
from utils import MapLogger
from utils.loss_functions import get_loss
from utils.file_handling import read_yaml_file
from models import Model


def main(args: ArgumentParser) -> None:
    """Main function for training and testing the model.

    Parameters:
        args: Command line arguments.
    """
    cwd = Path.cwd()
    run_config: dict = read_yaml_file(cwd / f"../configs/runs/{args.run_config}")
    wandb_config: dict = run_config["wandb"]
    dataset_config: dict = run_config["dataset"]
    model_config: dict = run_config["model"]
    model_config.update(
        read_yaml_file(cwd / f"../configs/model/{model_config['name']}.yaml")
    )
    model_config["batch_size"] = dataset_config["batch_size"]
    loss_config: dict = read_yaml_file(cwd / "../configs/loss_functions.yaml")
    trainer_config: dict = run_config["trainer"]

    if wandb_config["online"]:
        if not load_dotenv(cwd / "../.env"):
            raise FileNotFoundError(
                "Please add your WandB API key to .env in the root directory."
            )
        else:
            wandb.login(key=os.getenv("WANDB_API_KEY"))

    wandb_logger = WandbLogger(
        project=wandb_config["project"],
        log_model="all",
        offline=not wandb_config["online"],
        config=run_config,
        resume="must" if wandb_config["run"]["id"] is not None else None,
        id=wandb_config["run"]["id"],
    )

    datamodule = SRDDataModule(dataset_config)
    datamodule.prepare_data()
    rank_zero_info("Data module created.")

    loss = get_loss(loss_config)
    rank_zero_info(f"Using loss function: {loss}")

    prefix = "val" if args.stage == "train" else "test"
    metrics = get_metrics(prefix=prefix) if 1 == 0 else None  # TODO: Fix this
    metrics_on_inputs = (
        get_metrics_on_inputs(prefix=prefix) if 1 == 0 else None
    )  # TODO: Fix this
    extra_metrics = get_extra_metrics(prefix=prefix) if args.stage == "test" else None
    extra_metrics_on_inputs = (
        get_extra_metrics_on_inputs(prefix=prefix) if args.stage == "test" else None
    )

    if trainer_config["overfit_batches"] > 0:
        total_steps = trainer_config["overfit_batches"] * trainer_config["epochs"]
    else:
        total_steps = len(datamodule.train_dataloader()) * trainer_config["epochs"]

    model = Model(
        model_config,
        loss=loss,
        metrics=metrics,
        metrics_on_inputs=metrics_on_inputs,
        extra_metrics=extra_metrics,
        extra_metrics_on_inputs=extra_metrics_on_inputs,
        total_steps=total_steps,
    )

    wandb_logger.watch(model, log="all")

    callbacks = None

    if args.stage == "train":
        callbacks = []
        if trainer_config["log_maps_every_n_epochs"] > 0:
            map_logger = MapLogger(
                datamodule=datamodule,
                log_every_n_epochs=trainer_config["log_maps_every_n_epochs"],
            )
            callbacks.append(map_logger)
        model_checkpoint = ModelCheckpoint(
            monitor="val/loss",
            dirpath=f"{wandb_logger.experiment.dir}/checkpoints",
            filename=f"epoch:{{epoch:05d}}-val_loss:{{val/loss:.5f}}",
            mode="min",
            auto_insert_metric_name=False,
        )
        callbacks.append(model_checkpoint)

    trainer = Trainer(
        logger=wandb_logger,
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"] if args.stage == "train" else 1,
        max_epochs=trainer_config["epochs"],
        strategy=trainer_config["strategy"],
        callbacks=callbacks,
        benchmark=True,  # speeds up training if input is not changing size
        overfit_batches=trainer_config["overfit_batches"],
        log_every_n_steps=trainer_config["log_every_n_steps"],
    )

    if args.stage == "train":
        if trainer_config["checkpoint_path"] is not None:
            rank_zero_warn("A checkpoint_path is being used.")
        trainer.fit(
            model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=trainer_config["checkpoint_path"],
        )
    else:
        trainer.test(
            model, datamodule=datamodule, ckpt_path=trainer_config["checkpoint_path"]
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "stage", choices=["train", "test"], help="What stage to execute"
    )
    parser.add_argument(
        "run_config", type=Path, help="Yaml file name of the run config."
    )
    args = parser.parse_args()

    main(args)
