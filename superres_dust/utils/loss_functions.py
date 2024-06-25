from typing import Dict, Union

from torchmetrics import MeanAbsoluteError, Metric
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)

from metrics import VGGLoss


def get_loss(
    loss_config: Dict[str, Union[int, dict]],
) -> Metric:
    """Create the loss function as a weighted sum
    of different metrics.

    Parameters:
        loss_config: Dictionary containing the relative
            weightings of different metrics to be used
            in the loss function.

    Returns:
        Metric: Loss function.
    """
    l1_p = loss_config["l1"]
    ssim_p = loss_config["ssim"]
    ms_ssim_p = loss_config["ms_ssim"]

    vgg_config = loss_config["vgg"]
    vgg_p = vgg_config["p"]
    total = l1_p + ssim_p + ms_ssim_p + vgg_p
    if not 0.0 < total <= 1.0:
        raise ValueError(
            f"l1_p={l1_p} + ssim_p={ssim_p} + ms_ssim_p={ms_ssim_p} + "
            f"vgg_p={vgg_p} = {total}. Modify the loss configs such that "
            "they sum to a value in the range (0.0, 1.0]"
        )

    total_loss = 0

    if l1_p > 0.0:
        m = MeanAbsoluteError() * l1_p
        total_loss += m

    if ssim_p > 0.0:
        m = StructuralSimilarityIndexMeasure() * ssim_p
        total_loss += m

    if ms_ssim_p > 0.0:
        m = MultiScaleStructuralSimilarityIndexMeasure() * ms_ssim_p
        total_loss += m

    if vgg_p > 0.0:
        m = (
            VGGLoss(
                vgg_model=vgg_config["vgg_model"],
                batch_norm=vgg_config["batch_norm"],
                layers=vgg_config["layers"],
            )
            * vgg_p
        )
        total_loss += m

    return total_loss
