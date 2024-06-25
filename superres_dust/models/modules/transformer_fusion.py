import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp


def _patch_attention(m):
    """Patch the attention module to return the attention weights."""
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        """Wrap the forward method to return the attention weights."""
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class _SaveOutput:
    """Save the output of a module for logging."""

    def __init__(self):
        self.outputs = None

    def __call__(self, module, module_in, module_out):
        self.outputs = module_out[1]


class TransformerFusion(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        nhead: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        """Transformer-based fusion module.

        Parameters:
            in_channels: Number of channels in each of the encoded data.
            nhead: Number of attention heads.
            dropout: Dropout rate.
            num_layers: Number of transformer layers.
        """
        super().__init__()

        dim_feedforward = 4 * in_channels

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_layers
        )

        # Make the first and last layer attention weights available
        # for logging. They're not accessible by default.
        self.first_layer_attn = _SaveOutput()
        self.last_layer_attn = _SaveOutput()
        _patch_attention(self.encoder.layers[0].self_attn)
        _patch_attention(self.encoder.layers[-1].self_attn)
        self.encoder.layers[0].self_attn.register_forward_hook(self.first_layer_attn)
        self.encoder.layers[-1].self_attn.register_forward_hook(self.last_layer_attn)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return self.encoder(x)
