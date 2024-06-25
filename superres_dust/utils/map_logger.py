import pytorch_lightning as pl
import torch
import matplotlib as mpl
import matplotlib.colors as mcolors
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader


class MapLogger(pl.Callback):
    """Callback for logging images to WandB."""

    def __init__(self, datamodule: pl.LightningDataModule, log_every_n_epochs: int):
        super().__init__()
        self.data, self.resol = self._create_unique_resol_batch(
            datamodule.val_dataloader()
        )

        self.log_every_n_epochs = log_every_n_epochs
        self.names = [
            "tau",
            "q_hi",
            "u_hi",
            "q_lr",
            "u_lr",
            "q_hr",
            "u_hr",
            "q_pred",
            "u_pred",
        ]

    @staticmethod
    @rank_zero_only
    def _create_unique_resol_batch(
        dataloader: DataLoader, unique_resols: set = {15, 20, 25, 30}
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a batch of images with unique resolutions to evaluate
        the model's performance on different resolutions. This is run
        once at the beginning of the training.

        Parameters:
            dataloader: DataLoader containing the data.
            unique_resols: Set of unique resolutions to collect images for.

        Returns:
            Tuple containing the batch of images and their resolutions.
        """
        collected_images = []
        collected_resol = []

        for data, resol in dataloader:
            for idx in range(len(resol)):
                value = resol[idx].item()
                if value in unique_resols and value not in collected_resol:
                    collected_images.append(data[idx])
                    collected_resol.append(value)

                    # If we have collected all required images, we can create the batch
                    if len(collected_resol) == len(unique_resols):
                        new_data = torch.stack(collected_images)
                        new_resol = torch.tensor(collected_resol)
                        return new_data, new_resol

        raise ValueError("Could not find images for all unique resol values.")

    @rank_zero_only
    def _img_to_wandb(
        self,
        logger: WandbLogger,
        image: torch.Tensor,
        i: int,
        resol: torch.Tensor,
        stage: str,
    ) -> None:
        """Log an image to WandB.

        Parameters:
            logger: WandB logger.
            image: Image to log.
            i: Index of the image in the input tensor.
            resol: Resolution of the image.
            stage: Stage of the training (train, val, or test).
        """
        ext = "_encoded" if i > 8 else ""
        cm = "inferno" if i % 9 == 0 else "twilight"
        vmax = torch.max(torch.abs(image)).item()
        vmin = torch.min(torch.abs(image)).item() if i % 9 == 0 else -vmax
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        image = mpl.colormaps[cm](norm(image.squeeze().cpu().numpy()))
        key = f"{stage}/{self.names[i % 9]}_resol{int(resol)}{ext}"
        logger.log_image(key=key, images=[image])

    @rank_zero_only
    def _attn_to_wandb(
        self,
        logger: WandbLogger,
        image: torch.Tensor,
        layer: str,
        resol: torch.Tensor,
        stage: str,
    ) -> None:
        """Log an attention matrix to WandB.

        Parameters:
            logger: WandB logger.
            image: Attention matrix to log.
            layer: Attention layer (first or last).
            resol: Resolution of the image.
            stage: Stage of the training (train, val, or test).
        """
        vmax = torch.max(torch.abs(image)).item()
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        image = mpl.colormaps["inferno"](norm(image.squeeze().cpu().numpy()))
        key = f"{stage}/{layer}attn_h1qu_hnqu_resol{int(resol)}"
        logger.log_image(key=key, images=[image])

    @rank_zero_only
    def _log_images(
        self,
        logger: WandbLogger,
        pl_module: "pl.LightningModule",
        stage: str,
        first_epoch: bool = False,
    ) -> None:
        """Log images to WandB.

        Parameters:
            logger: WandB logger.
            pl_module: LightningModule containing the model.
            stage: Stage of the training (train, val, or test).
            first_epoch: Whether this is the first epoch.
        """
        self.data = self.data.to(pl_module.device)
        self.resol = self.resol.to(pl_module.device)
        preds = pl_module(self.data, self.resol)

        # 25600, 2, 6, 6
        first_layer_attn = pl_module.model.transformer_fusion.first_layer_attn.outputs
        last_layer_attn = pl_module.model.transformer_fusion.last_layer_attn.outputs[
            :, :, -2:, :
        ]
        nheads = last_layer_attn.shape[1]
        # 25600, 2, 6, 6 -> 4, 80*80, 2, 6, 6
        first_layer_attn = (
            first_layer_attn.view(self.data.shape[0], -1, nheads, 6, 6)
            .mean(1)  # over 80*80
            .view(-1, nheads * 6, 6)  # batch, nheads * 6, 6
        )
        last_layer_attn = (
            last_layer_attn.view(self.data.shape[0], -1, nheads, 2, 6)
            .mean(1)  # over 80*80
            .view(-1, nheads * 2, 6)
        )

        for j in range(4):
            # only log the input images once
            if first_epoch:
                for i in range(7):
                    self._img_to_wandb(
                        logger, self.data[j, i, ...], i, self.resol[j], stage
                    )

            for i in range(2):
                self._img_to_wandb(
                    logger, preds[j, i, ...], i + 7, self.resol[j], stage
                )

            # top left is index 0, 0
            self._attn_to_wandb(
                logger, first_layer_attn[j], "first", self.resol[j], stage
            )
            self._attn_to_wandb(
                logger, last_layer_attn[j], "last", self.resol[j], stage
            )

        for i in range(5):
            feats = pl_module.model.map_encoders[i](self.data[:, i])
            for j in range(4):
                for feat in feats[j]:
                    self._img_to_wandb(logger, feat, i + 9, self.resol[j], stage)

    @rank_zero_only
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Log images at the end of the validation loop.

        Parameters:
            trainer: Trainer object.
            pl_module: LightningModule containing the model.
        """
        if not trainer.sanity_checking:
            if trainer.current_epoch == 0:
                self._log_images(trainer.logger, pl_module, "val", True)
            if ((trainer.current_epoch + 1) % self.log_every_n_epochs) == 0:
                self._log_images(trainer.logger, pl_module, "val")

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Log images at the end of the test loop.

        Parameters:
            trainer: Trainer object.
            pl_module: LightningModule containing the model.
        """
        self._log_images(trainer.logger, pl_module, "test")
