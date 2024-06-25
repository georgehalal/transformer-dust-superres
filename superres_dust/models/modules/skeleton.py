import torch
from torch import nn

from .feature_extractor import MapFeatureExtractor
from .transformer_fusion import TransformerFusion


class Skeleton(nn.Module):
    """The full model connecting all the modules."""

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.resolution_embedding = nn.ModuleDict(
            {
                "emb": nn.Embedding(4, config["num_channels"]),
                "lin": nn.Sequential(
                    nn.Linear(1, config["num_channels"] * 4),
                    nn.GELU(),
                    nn.Linear(config["num_channels"] * 4, config["num_channels"]),
                ),
            }
        )

        self.map_encoders = nn.ModuleList(
            [
                MapFeatureExtractor(
                    in_channels=1,
                    out_channels=config["num_channels"],
                    num_channels=config["num_channels"],
                    num_res_blocks=config["num_res_blocks_encoder"],
                    memory_efficient=config["memory_efficient"],
                )
                for _ in range(5)
            ]
        )

        self.transformer_fusion = TransformerFusion(
            in_channels=config["num_channels"],
            nhead=config["nhead"],
            dropout=config["dropout"],
            num_layers=config["num_layers"],
        )

        self.map_decoders = nn.ModuleList(
            [
                MapFeatureExtractor(
                    in_channels=config["num_channels"],
                    out_channels=1,
                    num_channels=config["num_channels"],
                    num_res_blocks=config["num_res_blocks_decoder"],
                    memory_efficient=config["memory_efficient"],
                )
                for _ in range(2)
            ]
        )

    def forward(self, data: torch.Tensor, resol: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters:
            data: Input data tensor of shape (N, 5+, 1, H, W).
            resol: Resolution tensor of size N.

        Returns:
            torch.Tensor: Output tensor of shape (N, 2, 1, H, W).
        """
        # Embed the resolution
        if self.config["use_resolution_embedding"]:
            resol_emb = self.resolution_embedding["emb"](resol.int() // 5 - 3)
        else:
            resol_emb = self.resolution_embedding["lin"](resol)
        # N, Cout -> 1, N*H*W, Cout
        resol_emb = (
            resol_emb.unsqueeze(1)
            .repeat(1, data.shape[-2] * data.shape[-1], 1)
            .view(1, -1, resol_emb.shape[-1])
        )

        # Encode the input maps: tau, q_hi, u_hi, q_lr, u_lr
        maps_emb = torch.stack(
            [self.map_encoders[i](data[:, i, ...]) for i in range(5)]
        )

        # S, N, Cout, H, W -> S, N, H, W, Cout
        maps_emb = maps_emb.permute(0, 1, 3, 4, 2)
        # S, N, H, W, Cout -> S, N*H*W, Cout
        maps_emb = maps_emb.reshape(maps_emb.shape[0], -1, maps_emb.shape[-1])

        # merge resol_emb and maps_emb along the S dimension
        emb = torch.cat((resol_emb, maps_emb), 0)

        # Transformer fusion
        emb = self.transformer_fusion(emb)
        # S, N*H*W, Cout -> S, N, H, W, Cout -> S, N, Cout, H, W
        emb = emb.view(
            emb.shape[0], data.shape[0], data.shape[-2], data.shape[-1], emb.shape[-1]
        )
        emb = emb.permute(0, 1, 4, 2, 3)

        # Decode emb(q_lr, u_lr) -> q_hr, u_hr
        # N, 1, H, W -> N, 2, 1, H, W
        out = torch.stack(
            [self.map_decoders[i - 4](emb[i]) for i in range(4, 6)], dim=1
        )
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out
