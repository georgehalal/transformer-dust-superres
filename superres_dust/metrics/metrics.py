import piq
import torch
from torch.nn.functional import mse_loss
from torchmetrics import Metric
from torchvision import transforms
from torchvision.models import VGG, vgg


class CustomMetric(Metric):
    """
    A custom metric class that inherits from torchmetrics.Metric.
    It defines states that can be automatically synchronized
    across multiple devices and aggregated by summation.
    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "accumulated_metric",
            default=torch.tensor(0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sample_count",
            default=torch.tensor(0, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str
    ) -> None:
        """
        Update the metric with the given predictions and targets.
        To be implemented by subclasses.

        Parameters:
            predictions: Predictions from the model.
            target: Ground truth values.
            reduction: Reduction method for the metric.
        """
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric value.

        Returns:
            The computed metric value
        """
        return self.accumulated_metric / self.sample_count


class VIF(CustomMetric):
    """
    Compute Visual Information Fidelity in pixel domain for a batch of images.
    """

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.vif_p(x=predictions, y=target, reduction=reduction)
        self.total += predictions.size()[0]


class MDSI(CustomMetric):
    """
    Compute Mean Deviation Similarity Index (MDSI) for a batch of images.
    """

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.mdsi(x=predictions, y=target, reduction=reduction)
        self.total += predictions.size()[0]


class HaarPSI(CustomMetric):
    """
    Compute Haar Wavelet-Based Perceptual Similarity Inputs
    """

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.haarpsi(x=predictions, y=target, reduction=reduction)
        self.total += predictions.size()[0]


class GMSD(CustomMetric):
    """
    Compute Gradient Magnitude Similarity Deviation
    """

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.gmsd(x=predictions, y=target, reduction=reduction)
        self.total += predictions.size()[0]


class MultiScaleGMSD(CustomMetric):
    """
    Computation of Multi scale GMSD.
    """

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.multi_scale_gmsd(
            x=predictions, y=target, chromatic=False, reduction=reduction
        )
        self.total += predictions.size()[0]


class FSIM(CustomMetric):
    """
    Compute Feature Similarity Index Measure for a batch of images.
    """

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        self.metric += piq.fsim(
            x=predictions, y=target, chromatic=False, reduction=reduction
        )
        self.total += predictions.size()[0]


class VGGLoss(CustomMetric):
    def __init__(
        self,
        vgg_model: str = "vgg19",
        batch_norm: bool = False,
        layers: int = 8,
    ):
        """
        Initialize the VGG loss metric.

        Parameters:
            vgg_model: The VGG model to use.
            batch_norm: Whether to use batch normalization.
            layers: The number of layers to use.
        """
        super().__init__()

        if vgg_model not in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            raise ValueError(f"Unknown vgg-model: {vgg_model}")

        if batch_norm:
            vgg_model = f"{vgg_model}_bn"

        models = {
            "vgg11": vgg.vgg11,
            "vgg11_bn": vgg.vgg11_bn,
            "vgg13": vgg.vgg13,
            "vgg13_bn": vgg.vgg13_bn,
            "vgg16": vgg.vgg16,
            "vgg16_bn": vgg.vgg16_bn,
            "vgg19": vgg.vgg19,
            "vgg19_bn": vgg.vgg19_bn,
        }

        # mean and std come from ImageNet dataset since VGG is trained on that data
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.m: VGG = models[vgg_model]().features[: layers + 1]
        self.m.eval()
        self.m.requires_grad_(False)

    def update(
        self, predictions: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> None:
        predictions = predictions.repeat(1, 1, 3, 1, 1)
        target = target.repeat(1, 1, 3, 1, 1)
        q_pred = predictions[:, 0, ...]
        u_pred = predictions[:, 1, ...]
        q_target = target[:, 0, ...]
        u_target = target[:, 1, ...]

        q_pred = self.m(self.normalize(q_pred))
        u_pred = self.m(self.normalize(u_pred))
        q_target = self.m(self.normalize(q_target))
        u_target = self.m(self.normalize(u_target))

        self.metric += mse_loss(input=q_pred, target=q_target, reduction=reduction)
        self.metric += mse_loss(input=u_pred, target=u_target, reduction=reduction)
        self.total += q_pred.size()[0] + u_pred.size()[0]
