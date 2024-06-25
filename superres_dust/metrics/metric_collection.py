from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)

from metrics import (
    FSIM,
    GMSD,
    MDSI,
    VIF,
    HaarPSI,
    MultiScaleGMSD,
)


def get_metrics(prefix: str) -> MetricCollection:
    """Metrics to log during validation and testing
    between the output of the model and the target.

    Parameters:
        prefix: Prefix to add to the metric names.

    Returns:
        MetricCollection: Collection of metrics.
    """
    return MetricCollection(
        {
            "ssim": StructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "l1": MeanAbsoluteError(),
            "l2": MeanSquaredError(),
        },
        prefix=f"{prefix}/",
    )


def get_extra_metrics(prefix: str) -> MetricCollection:
    """Metrics to log during testing only
    between the output of the model and the target.

    Parameters:
        prefix: Prefix to add to the metric names.

    Returns:
        MetricCollection: Collection of metrics.
    """
    return MetricCollection(
        {
            "vif_p": VIF(),
            "fsim": FSIM(),
            "gmsd": GMSD(),
            "ms_gmsd": MultiScaleGMSD(),
            "haarpsi": HaarPSI(),
            "mdsi": MDSI(),
        },
        prefix=f"{prefix}/",
    )


def get_metrics_on_inputs(prefix: str) -> MetricCollection:
    """Metrics to log during validation and testing
    between the input to the model and the target.

    Parameters:
        prefix: Prefix to add to the metric names.

    Returns:
        MetricCollection: Collection of metrics.
    """
    return MetricCollection(
        {
            "in/ssim": StructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "in/ms_ssim": MultiScaleStructuralSimilarityIndexMeasure(
                kernel_size=13, sigma=2.5, k2=0.05
            ),
            "in/l1": MeanAbsoluteError(),
            "in/l2": MeanSquaredError(),
        },
        prefix=f"{prefix}/",
    )


def get_extra_metrics_on_inputs(prefix: str) -> MetricCollection:
    """Metrics to log during testing only
    between the input to the model and the target.

    Parameters:
        prefix: Prefix to add to the metric names.

    Returns:
        MetricCollection: Collection of metrics.
    """
    return MetricCollection(
        {
            "in/vif_p": VIF(),
            "in/fsim": FSIM(),
            "in/gmsd": GMSD(),
            "in/ms_gmsd": MultiScaleGMSD(),
            "in/haarpsi": HaarPSI(),
            "in/mdsi": MDSI(),
        },
        prefix=f"{prefix}/",
    )
