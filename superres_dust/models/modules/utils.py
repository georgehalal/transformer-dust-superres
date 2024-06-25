import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp


def make_layer(block: nn.Module, n_layers: int) -> nn.Sequential:
    """Repeat a block n_layers times.

    Parameters:
        block: The block to be repeated.
        n_layers: Number of times to repeat the block.

    Returns:
        A sequential container with the repeated blocks.
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def apply_layer(in_tensors: list[torch.Tensor], layer: nn.Module) -> torch.Tensor:
    """The checkpoint function is used to reduce the memory usage during the
    forward pass of the model. It allows the computation to be performed
    in chunks, and the intermediate activations are not stored for the
    backward pass. Instead, during the backward pass, the forward
    computation is performed again to reconstruct the necessary
    intermediate activations. This trade-off between memory usage and
    computation time can be beneficial when dealing with large models or
    limited memory resources.

    The use_reentrant=False argument indicates that torch.cat cannot
    be called recursively. This is a hint to the checkpoint function
    to optimize the computation accordingly.

    It's important to note that using checkpoint adds some computational
    overhead due to the re-computation of the forward pass during the
    backward pass. Therefore, it should be used judiciously and in cases
    where the memory savings outweigh the computational cost.

    Parameters:
        in_tensors: List of input tensors.
        layer: The layer to apply to the input tensors.

    Returns:
        torch.Tensor: Output tensor after applying the layer
    """
    concat = cp(torch.cat, tensors=in_tensors, dim=1, use_reentrant=False)
    return layer(concat)
