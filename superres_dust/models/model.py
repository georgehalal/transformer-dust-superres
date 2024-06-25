from typing import Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from .modules import Skeleton


class Model(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        loss: Optional[Metric],
        metrics: Optional[MetricCollection],
        extra_metrics: Optional[MetricCollection],
        metrics_on_inputs: Optional[MetricCollection],
        extra_metrics_on_inputs: Optional[MetricCollection],
        total_steps: int,
    ):
        """
        Parameters:
            config: Dictionary containing the hyperparameters
                of the model.
            loss: Loss function to use.
            metrics: Metrics to log during validation and testing
                between the output of the model and the target.
            extra_metrics: Metrics to log during testing only
                between the output of the model and the target.
            metrics_on_inputs: Metrics to log during validation and testing
                 between the input to the model and the target.
            extra_metrics_on_inputs: Metrics to log during testing only
                between the input to the model and the target.
            total_steps: Total number of steps in the training loop
                used for the learning rate scheduler.
        """
        super(Model, self).__init__()

        # metrics to log if any
        self.metrics = metrics
        self.extra_metrics = extra_metrics
        self.metrics_on_inputs = metrics_on_inputs
        self.extra_metrics_on_inputs = extra_metrics_on_inputs

        # used for learning rate scheduler
        self.total_steps = total_steps

        # Optimizer parameters
        self.learning_rate = config["learning_rate"]
        self.betas = (config["b1"], config["b2"])

        self.loss = loss
        self.batch_size = config["batch_size"]

        self.model = Skeleton(config=config)

        # log hyperparameters
        self.save_hyperparameters(ignore=["loss"])

    def forward(self, data: Tensor, resol: Tensor) -> torch.Tensor:
        return self.model(data, resol)

    def training_step(self, batch: Tensor, batch_idx: int) -> Optional[Tensor]:
        return self._on_step(batch, "train")

    def on_validation_start(self) -> None:
        self._on_epoch_end("train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Optional[Tensor]:
        self._on_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end("val")

    def test_step(self, batch: Tensor, batch_idx: int) -> Optional[Tensor]:
        self._on_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end("test")

    def on_after_backward(self) -> None:
        """Log the learning rate after each step.
        Useful when using a learning rate scheduler.
        """
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def _on_step(self, batch: Tensor, stage: str) -> Optional[Tensor]:
        """Perform a step in the train or validation loop.

        Parameters:
            batch: The batch of data.
            stage: train, val, or test.
        """
        data, resol = batch
        preds = self(data, resol)
        target = data[:, -2:, ...]

        if stage == "train":
            loss = self.loss(preds=preds, target=target)
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=self.batch_size,
                on_step=True,
                on_epoch=False,
            )
            return loss
        else:
            self.loss.update(preds=preds, target=target)

            if self.metrics is not None:
                self.metrics.update(preds=preds, target=target)

            if self.metrics_on_inputs is not None:
                self.metrics_on_inputs.update(preds=data[:, -4:-2, ...], target=target)

            if self.extra_metrics is not None:
                self.extra_metrics.update(preds=preds, target=target)

            if self.extra_metrics_on_inputs is not None:
                self.extra_metrics_on_inputs.update(
                    preds=data[:, -4:-2, ...], target=target
                )

    def _on_epoch_end(self, stage: str) -> None:
        """Log the metrics (and loss) at the end of an epoch.

        Parameters:
            stage: train, val, or test.
        """
        if stage == "train":
            self.loss.reset()
        else:
            loss = self.loss.compute()
            self.log(
                f"{stage}/loss",
                loss,
                batch_size=self.batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.loss.reset()

            to_log: list[dict] = []

            if self.metrics is not None:
                to_log.append(self.metrics.compute())
                self.metrics.reset()

            if self.extra_metrics is not None:
                to_log.append(self.extra_metrics.compute())
                self.extra_metrics.reset()

            if self.metrics_on_inputs is not None:
                to_log.append(self.metrics_on_inputs.compute())
                self.metrics_on_inputs.reset()
                if not self.trainer.sanity_checking:
                    self.metrics_on_inputs = None

            if self.extra_metrics_on_inputs is not None:
                to_log.append(self.extra_metrics_on_inputs.compute())
                self.extra_metrics_on_inputs.reset()
                if not self.trainer.sanity_checking:
                    self.extra_metrics_on_inputs = None

            for log in to_log:
                self.log_dict(
                    log,
                    batch_size=self.batch_size,
                    on_step=False,
                    on_epoch=True,
                    # across devices. may lead to significant overhead:
                    sync_dist=True,
                )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Define the optimizer. We use AdamW in this case.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=0.01,
        )
        return optimizer
