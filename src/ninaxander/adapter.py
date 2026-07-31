"""Shared-latent adapter architecture used by training and every evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-norm residual MLP block at constant width."""

    def __init__(self, width: int, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden_size)
        self.fc2 = nn.Linear(hidden_size, width)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.fc2(F.gelu(self.fc1(self.norm(value))))


def _resnet(
    input_size: int,
    output_size: int,
    width: int,
    blocks: int,
    hidden_size: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, width),
        *(ResBlock(width, hidden_size) for _ in range(blocks)),
        nn.Linear(width, output_size),
    )


def _plain_mlp(input_size: int, output_size: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, output_size),
    )


class LatentAdapter(nn.Module):
    """Bidirectional shared-latent residual adapter."""

    def __init__(
        self,
        d_a: int,
        d_b: int,
        latent_size: int,
        blocks: int = 2,
        hidden: int = 1024,
        arch: str = "resnet",
    ) -> None:
        super().__init__()
        if arch == "mlp":
            def make(input_size, output_size):
                return _plain_mlp(input_size, output_size, hidden)
        elif arch == "resnet":
            def make(input_size, output_size):
                return _resnet(
                    input_size,
                    output_size,
                    latent_size,
                    blocks,
                    hidden,
                )
        else:
            raise ValueError(f"Unknown adapter architecture: {arch}")
        self.eA = make(d_a, latent_size)
        self.eB = make(d_b, latent_size)
        self.norm = nn.LayerNorm(latent_size, elementwise_affine=False)
        self.dA = make(latent_size, d_a)
        self.dB = make(latent_size, d_b)

    def enc_a(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.eA(value))

    def enc_b(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.eB(value))

    def forward(
        self,
        value_a: torch.Tensor,
        value_b: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_a = self.enc_a(value_a)
        latent_b = self.enc_b(value_b)
        decode_a = latent_a + sigma * torch.randn_like(latent_a) if sigma > 0 else latent_a
        decode_b = latent_b + sigma * torch.randn_like(latent_b) if sigma > 0 else latent_b
        return latent_a, latent_b, self.dA(decode_a), self.dB(decode_b)


def r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction - target).pow(2).sum()
    denominator = (target - target.mean(0)).pow(2).sum() + 1e-9
    return (1 - numerator / denominator).item()


@torch.no_grad()
def blocks_of(model: nn.Module) -> nn.ModuleList:
    """Return the block list for either supported parent architecture."""

    if hasattr(model, "rwkv"):
        return model.rwkv.blocks
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    raise ValueError(f"Unknown parent architecture: {type(model)}")
