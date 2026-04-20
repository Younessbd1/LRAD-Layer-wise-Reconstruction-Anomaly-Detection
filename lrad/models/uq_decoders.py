"""Uncertainty-aware decoders for LRAD.

Three UQ strategies built on top of the base decoders:

1. MCDropoutCNNDecoder - Adds dropout layers that remain active at test time.
   Multiple forward passes -> distribution of reconstructions -> variance = uncertainty.

2. MCDropoutMLPDecoder - Same principle for MLP decoders.

3. EnsembleDecoder - Wraps M independently-trained decoders. Disagreement = uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MCDropoutCNNDecoder(nn.Module):
    """CNN decoder with MC Dropout for uncertainty quantification.

    Dropout is placed AFTER each ConvTranspose2d + BN + ReLU block.
    At test time, dropout stays active -> multiple passes give different
    reconstructions -> variance captures epistemic uncertainty.

    Args:
        channels: Channel dimensions from input to decoded layer.
        spatial_sizes: Spatial sizes at each level.
        dropout_rate: Dropout probability (default 0.2).
    """

    def __init__(
        self,
        channels: list[int],
        spatial_sizes: list[int],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        assert len(channels) == len(spatial_sizes)

        KERNEL, STRIDE, PADDING = 3, 2, 1
        rev_ch = channels[::-1]
        rev_sz = spatial_sizes[::-1]

        self.layers = nn.ModuleList()
        self.dropout_rate = dropout_rate

        for i in range(len(rev_ch) - 1):
            is_last = (i == len(rev_ch) - 2)
            raw_out = (rev_sz[i] - 1) * STRIDE - 2 * PADDING + KERNEL
            out_pad = 1 if rev_sz[i + 1] != raw_out else 0

            self.layers.append(nn.ConvTranspose2d(
                rev_ch[i], rev_ch[i + 1],
                kernel_size=KERNEL, stride=STRIDE,
                padding=PADDING, output_padding=out_pad,
            ))

            if not is_last:
                self.layers.append(nn.BatchNorm2d(rev_ch[i + 1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout2d(dropout_rate))
            else:
                self.layers.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def mc_forward(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
    ) -> dict:
        """Run T stochastic forward passes and compute statistics.

        Args:
            x: Input activation (B, C, H, W).
            n_samples: Number of MC samples (T).

        Returns:
            dict with:
                'mean': (B, C_out, H_out, W_out) mean reconstruction
                'variance': (B, 1, H_out, W_out) pixel-wise variance (uncertainty)
                'samples': (T, B, C_out, H_out, W_out) all samples
        """
        self.train()  # Keep dropout active
        samples = []

        for _ in range(n_samples):
            with torch.no_grad():
                recon = self.forward(x)
                samples.append(recon)

        samples = torch.stack(samples, dim=0)  # (T, B, C, H, W)
        mean = samples.mean(dim=0)             # (B, C, H, W)
        variance = samples.var(dim=0).mean(dim=1, keepdim=True)  # (B, 1, H, W)

        return {
            "mean": mean,
            "variance": variance,
            "samples": samples,
        }


class MCDropoutMLPDecoder(nn.Module):
    """MLP decoder with MC Dropout for uncertainty quantification.

    Args:
        activation_dim: Dimension of hidden activation vector.
        output_dim: Flattened image dimension.
        hidden_dims: Intermediate decoder widths.
        output_shape: (C, H, W) for reshape.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        activation_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        output_shape: tuple[int, int, int] = (1, 28, 28),
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.dropout_rate = dropout_rate

        if hidden_dims is None:
            hidden_dims = []

        dims = [activation_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if not is_last:
                self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_rate))
            else:
                self.layers.append(nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.view(-1, *self.output_shape)

    def mc_forward(self, x: torch.Tensor, n_samples: int = 20) -> dict:
        self.train()
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                recon = self.forward(x)
                samples.append(recon)

        samples = torch.stack(samples, dim=0)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0).mean(dim=1, keepdim=True)

        return {"mean": mean, "variance": variance, "samples": samples}


class EnsembleDecoder(nn.Module):
    """Ensemble of M decoders for uncertainty via disagreement.

    Each member is an independently-initialized decoder. At test time,
    the variance across member reconstructions captures epistemic uncertainty.

    Args:
        decoder_class: The base decoder class (CNNDecoder or MLPDecoder).
        n_members: Number of ensemble members.
        **decoder_kwargs: Arguments passed to each decoder constructor.
    """

    def __init__(
        self,
        decoder_class: type,
        n_members: int = 5,
        **decoder_kwargs,
    ):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList([
            decoder_class(**decoder_kwargs) for _ in range(n_members)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean reconstruction across ensemble members."""
        recons = [m(x) for m in self.members]
        return torch.stack(recons).mean(dim=0)

    def ensemble_forward(self, x: torch.Tensor) -> dict:
        """Run all members and compute statistics.

        Returns:
            dict with:
                'mean': (B, C, H, W) mean reconstruction
                'variance': (B, 1, H, W) pixel-wise variance (uncertainty)
                'member_outputs': list of (B, C, H, W) per-member outputs
        """
        with torch.no_grad():
            member_outputs = [m(x) for m in self.members]

        stacked = torch.stack(member_outputs, dim=0)  # (M, B, C, H, W)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0).mean(dim=1, keepdim=True)

        return {
            "mean": mean,
            "variance": variance,
            "member_outputs": member_outputs,
        }

    def get_optimizers(self, lr: float = 1e-3) -> list:
        """Return one optimizer per ensemble member for independent training."""
        return [torch.optim.Adam(m.parameters(), lr=lr) for m in self.members]
