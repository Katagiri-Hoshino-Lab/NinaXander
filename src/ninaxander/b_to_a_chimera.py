"""Trusted B-to-A (Pythia-prefix -> RWKV-suffix) execution utilities.

The adapter supports the four directional paths A-to-A, A-to-B, B-to-B, and
B-to-A, where A is RWKV and B is Pythia. This module centralizes the B-to-A
cross-family path so QA, generation, robustness, and serving benchmarks cannot
drift onto subtly different residual boundaries.

Layer ``L`` always means "after parent block L". A B-to-A chimera reads Pythia
``hidden_states[L + off_b]`` and replaces the input of RWKV block ``L + 1``.
The parent-reproduction gate in :meth:`gate_parent_reproduction` must pass
before any result is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .adapter import LatentAdapter


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Return the fp32 relative L2 error used by every execution gate."""

    return (
        (value - reference).float().norm().item()
        / (reference.float().norm().item() + 1e-9)
    )


@dataclass(frozen=True)
class ResidualOffsets:
    """Indices mapping a post-block layer number to HF ``hidden_states``."""

    rwkv: int
    pythia: int


@torch.no_grad()
def detect_residual_offsets(
    rwkv: torch.nn.Module,
    pythia: torch.nn.Module,
    probe_ids: torch.Tensor,
    dev_a: str | torch.device,
    dev_b: str | torch.device,
) -> ResidualOffsets:
    """Detect whether each HF model includes embeddings at hidden-state index 0."""

    ids_a = probe_ids.to(dev_a)
    ids_b = probe_ids.to(dev_b)
    out_a = rwkv(ids_a, output_hidden_states=True, use_cache=False)
    out_b = pythia(ids_b, output_hidden_states=True, use_cache=False)
    emb_a = rwkv.rwkv.embeddings(ids_a)
    emb_b = pythia.gpt_neox.embed_in(ids_b)
    off_a = 1 if (out_a.hidden_states[0] - emb_a).abs().max().item() < 1e-3 else 0
    off_b = 1 if (out_b.hidden_states[0] - emb_b).abs().max().item() < 1e-3 else 0
    return ResidualOffsets(rwkv=off_a, pythia=off_b)


class BToAChimera:
    """Execute Pythia through block L, translate B-to-A, then run RWKV's suffix."""

    def __init__(
        self,
        rwkv: torch.nn.Module,
        pythia: torch.nn.Module,
        adapter: LatentAdapter,
        stats: dict[int, list[torch.Tensor]],
        offsets: ResidualOffsets,
        dev_a: str | torch.device,
        dev_b: str | torch.device,
    ) -> None:
        self.rwkv = rwkv
        self.pythia = pythia
        self.adapter = adapter
        self.stats = stats
        self.offsets = offsets
        self.dev_a = torch.device(dev_a)
        self.dev_b = torch.device(dev_b)
        self._injected: torch.Tensor | None = None
        self._adapter_dtype = next(adapter.parameters()).dtype
        self._suffix_streams: dict[int, list[torch.cuda.Stream]] = {}

    def standardize_a(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        mean_a, std_a = self.stats[layer][0:2]
        return (hidden.to(self.dev_a).float() - mean_a) / std_a

    def standardize_b(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        mean_b, std_b = self.stats[layer][2:4]
        return (hidden.to(self.dev_a).float() - mean_b) / std_b

    def destandardize_a(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        mean_a, std_a = self.stats[layer][0:2]
        return (hidden.float() * std_a + mean_a).to(self.rwkv.dtype)

    @torch.no_grad()
    def translate_adapter(self, hidden_b: torch.Tensor, layer: int) -> torch.Tensor:
        """Translate a standardized Pythia residual through D_A(E_B(.)) ."""

        value_b = self.standardize_b(hidden_b, layer).to(self._adapter_dtype)
        decoded_a = self.adapter.dA(self.adapter.enc_b(value_b))
        return self.destandardize_a(decoded_a, layer)

    @torch.no_grad()
    def translate_self_a(self, hidden_a: torch.Tensor, layer: int) -> torch.Tensor:
        """A->A decoder control: reconstruction loss without cross-family transfer."""

        value_a = self.standardize_a(hidden_a, layer).to(self._adapter_dtype)
        decoded_a = self.adapter.dA(self.adapter.enc_a(value_a))
        return self.destandardize_a(decoded_a, layer)

    @torch.no_grad()
    def translate_linear(
        self,
        hidden_b: torch.Tensor,
        layer: int,
        linear_map: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a separately fitted affine standardized B->A map."""

        width = self.stats[layer][0].numel()
        value_b = self.standardize_b(hidden_b, layer).reshape(-1, width)
        design = torch.cat(
            [
                value_b,
                torch.ones(value_b.shape[0], 1, device=self.dev_a),
            ],
            dim=1,
        )
        decoded_a = (design @ linear_map).reshape(*hidden_b.shape[:-1], width)
        return self.destandardize_a(decoded_a, layer)

    def _prehook(self, module, args, kwargs):
        del module
        if self._injected is None:
            return None
        if args:
            return (self._injected,) + args[1:], kwargs
        replacement = dict(kwargs)
        # transformers.RwkvBlock.forward names its first argument ``hidden``.
        replacement["hidden"] = self._injected
        return args, replacement

    @torch.no_grad()
    def run_rwkv_suffix(
        self,
        input_ids: torch.Tensor,
        hidden_a: torch.Tensor,
        layer: int,
        **model_kwargs,
    ):
        """Inject ``hidden_a`` before RWKV block L+1 and return the HF output."""

        block_index = layer + 1
        if not 0 <= block_index < len(self.rwkv.rwkv.blocks):
            raise ValueError(
                f"B-to-A switch L={layer} has no RWKV suffix block L+1 "
                f"(model has {len(self.rwkv.rwkv.blocks)} blocks)"
            )
        if self._injected is not None:
            raise RuntimeError("nested B-to-A chimera injection is not supported")
        handle = self.rwkv.rwkv.blocks[block_index].register_forward_pre_hook(
            self._prehook,
            with_kwargs=True,
        )
        self._injected = hidden_a.to(self.dev_a)
        try:
            kwargs = dict(model_kwargs)
            kwargs.setdefault("use_cache", False)
            return self.rwkv(input_ids.to(self.dev_a), **kwargs)
        finally:
            self._injected = None
            handle.remove()

    @torch.no_grad()
    def run_rwkv_suffix_direct(
        self,
        hidden_a: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Run only RWKV blocks L+1..end, then final norm and LM head.

        This is identical to the computation after the injection point in
        :meth:`run_rwkv_suffix`, but avoids recomputing the discarded prefix.
        HF's original block indices are retained for its /2 rescale schedule.
        """

        first_block = layer + 1
        block_count = len(self.rwkv.rwkv.blocks)
        if not 0 <= first_block < block_count:
            raise ValueError(
                f"B-to-A switch L={layer} has no RWKV suffix "
                f"(model has {block_count} blocks)"
            )
        value = hidden_a.to(self.dev_a)
        state = None
        for original_index in range(first_block, block_count):
            value, state, _ = self.rwkv.rwkv.blocks[original_index](
                value,
                state=state,
                use_cache=False,
                output_attentions=False,
            )
            if (
                self.rwkv.rwkv.layers_are_rescaled
                and self.rwkv.config.rescale_every > 0
                and (original_index + 1) % self.rwkv.config.rescale_every == 0
            ):
                value = value / 2
        return self.rwkv.head(self.rwkv.rwkv.ln_out(value))

    @torch.no_grad()
    def run_rwkv_suffixes_concurrent(
        self,
        hidden_values: list[torch.Tensor] | tuple[torch.Tensor, ...],
        layer: int,
    ) -> list[torch.Tensor]:
        """Run independent batch-1 suffixes on separate CUDA streams.

        Unlike concatenating samples along the batch axis, this preserves the
        exact batch-1 RWKV kernel path. The caller must gate the results against
        sequential execution before relying on concurrency.
        """

        values = list(hidden_values)
        if len(values) <= 1 or self.dev_a.type != "cuda":
            return [
                self.run_rwkv_suffix_direct(value, layer)
                for value in values
            ]
        if len(values) not in self._suffix_streams:
            self._suffix_streams[len(values)] = [
                torch.cuda.Stream(device=self.dev_a)
                for _ in values
            ]
        streams = self._suffix_streams[len(values)]
        caller_stream = torch.cuda.current_stream(self.dev_a)
        outputs: list[torch.Tensor] = []
        for stream, value in zip(streams, values):
            stream.wait_stream(caller_stream)
            with torch.cuda.stream(stream):
                outputs.append(self.run_rwkv_suffix_direct(value, layer))
        for stream in streams:
            caller_stream.wait_stream(stream)
        return outputs

    @torch.no_grad()
    def gate_parent_reproduction(
        self,
        probe_ids: torch.Tensor,
        switches: Iterable[int],
        tolerance: float = 1e-4,
    ) -> dict[int, float]:
        """Require exact RWKV recovery when its true residual is re-injected."""

        output = self.rwkv(
            probe_ids.to(self.dev_a),
            output_hidden_states=True,
            use_cache=False,
        )
        errors: dict[int, float] = {}
        for layer in switches:
            hidden_a = output.hidden_states[layer + self.offsets.rwkv]
            reproduced = self.run_rwkv_suffix(
                probe_ids,
                hidden_a,
                layer,
            ).logits
            error = relative_error(reproduced, output.logits)
            if error >= tolerance:
                raise AssertionError(
                    f"B-to-A parent-reproduction gate failed at L={layer}: "
                    f"relative error={error:.3e}, tolerance={tolerance:.3e}"
                )
            errors[layer] = error
        return errors

    @torch.no_grad()
    def gate_direct_suffix(
        self,
        probe_ids: torch.Tensor,
        switches: Iterable[int],
        tolerance: float = 1e-4,
    ) -> dict[int, float]:
        """Require the prefix-free suffix path to reproduce pure RWKV logits."""

        output = self.rwkv(
            probe_ids.to(self.dev_a),
            output_hidden_states=True,
            use_cache=False,
        )
        errors: dict[int, float] = {}
        for layer in switches:
            hidden_a = output.hidden_states[layer + self.offsets.rwkv]
            reproduced = self.run_rwkv_suffix_direct(hidden_a, layer)
            error = relative_error(reproduced, output.logits)
            if error >= tolerance:
                raise AssertionError(
                    f"direct RWKV suffix gate failed at L={layer}: "
                    f"relative error={error:.3e}, tolerance={tolerance:.3e}"
                )
            errors[layer] = error
        return errors


@torch.no_grad()
def fit_b_to_a_affine_maps(
    runtime: BToAChimera,
    fit_ids: torch.Tensor,
    switches: Iterable[int],
    ridge: float = 1e-3,
    batch_windows: int = 4,
) -> dict[int, torch.Tensor]:
    """Fit affine standardized Pythia->RWKV maps with fp64 normal equations."""

    layers = list(switches)
    width_a = runtime.stats[layers[0]][0].numel()
    width_b = runtime.stats[layers[0]][2].numel()
    xtx = {
        layer: torch.zeros(width_b + 1, width_b + 1, dtype=torch.float64)
        for layer in layers
    }
    xty = {
        layer: torch.zeros(width_b + 1, width_a, dtype=torch.float64)
        for layer in layers
    }
    for start in range(0, fit_ids.shape[0], batch_windows):
        ids = fit_ids[start : start + batch_windows]
        output_a = runtime.rwkv(
            ids.to(runtime.dev_a),
            output_hidden_states=True,
            use_cache=False,
        )
        output_b = runtime.pythia(
            ids.to(runtime.dev_b),
            output_hidden_states=True,
            use_cache=False,
        )
        for layer in layers:
            value_b = runtime.standardize_b(
                output_b.hidden_states[layer + runtime.offsets.pythia],
                layer,
            ).reshape(-1, width_b).double()
            value_a = runtime.standardize_a(
                output_a.hidden_states[layer + runtime.offsets.rwkv],
                layer,
            ).reshape(-1, width_a).double()
            design = torch.cat(
                [
                    value_b,
                    torch.ones(
                        value_b.shape[0],
                        1,
                        dtype=torch.float64,
                        device=runtime.dev_a,
                    ),
                ],
                dim=1,
            )
            xtx[layer] += (design.T @ design).cpu()
            xty[layer] += (design.T @ value_a).cpu()
        if start == 0 or (start // batch_windows + 1) % 16 == 0:
            print(
                f"B-to-A affine-fit windows "
                f"{min(start + batch_windows, fit_ids.shape[0])}/{fit_ids.shape[0]}",
                flush=True,
            )

    maps: dict[int, torch.Tensor] = {}
    for layer in layers:
        scale = torch.diagonal(xtx[layer]).sum() / (width_b + 1)
        regularized = xtx[layer] + ridge * scale * torch.eye(
            width_b + 1,
            dtype=torch.float64,
        )
        maps[layer] = torch.linalg.solve(regularized, xty[layer]).float().to(runtime.dev_a)
    return maps
