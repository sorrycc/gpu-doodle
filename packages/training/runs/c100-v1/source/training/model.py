"""The sequence classifier used by gpu-doodle, adapted from gpu-time's TimeTagger.

Inputs are the eight per-point features from `dataset.py`. The output is one
logit per class for the whole sketch: there is no per-point head and no
boundary head. The affine scan, fake quantization, and storage rounding are
the same as gpu-time so the exporter and the WGSL kernel can follow its shape.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FEATURES = 8
HIDDEN = 64
HEAD = 64
CLASSES = 100
CONV_WIDTH = 5


def affine_scan(gate: Tensor, candidate: Tensor) -> Tensor:
    """Inclusive parallel scan for state[t] = gate[t] * state[t-1] + candidate[t]."""
    width = gate.shape[1]
    stride = 1
    while stride < width:
        next_gate = gate[:, stride:] * gate[:, :-stride]
        next_candidate = candidate[:, stride:] + gate[:, stride:] * candidate[:, :-stride]
        gate = torch.cat((gate[:, :stride], next_gate), dim=1)
        candidate = torch.cat((candidate[:, :stride], next_candidate), dim=1)
        stride *= 2
    return candidate


def sequential_scan(gate: Tensor, candidate: Tensor) -> Tensor:
    """The same recurrence evaluated in order; the CPU and WGSL references use this."""
    state = torch.zeros_like(candidate[:, 0])
    states = []
    for index in range(gate.shape[1]):
        state = gate[:, index] * state + candidate[:, index]
        states.append(state)
    return torch.stack(states, dim=1)


def quantize(weight: Tensor, bits: int = 6) -> Tensor:
    """Symmetric per-tensor fake quantization with a straight-through gradient."""
    maximum = (1 << (bits - 1)) - 1
    scale = (weight.detach().abs().max() / maximum).clamp_min(1e-8)
    quantized = (weight / scale).round().clamp(-maximum, maximum) * scale
    return weight + (quantized - weight).detach()


def half_storage(value: Tensor) -> Tensor:
    rounded = value.half().float()
    return value + (rounded - value).detach()


class DoodleTagger(nn.Module):
    def __init__(self, classes: int = CLASSES) -> None:
        super().__init__()
        self.classes = classes
        self.input_weight = nn.Parameter(torch.empty(HIDDEN, FEATURES))
        self.input_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.convolution = nn.Parameter(torch.empty(CONV_WIDTH, HIDDEN))
        self.encoder_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.gate_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.gate_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.candidate_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.candidate_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.combine_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN * 2))
        self.combine_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.head_weight = nn.Parameter(torch.empty(HEAD, HIDDEN * 2))
        self.head_bias = nn.Parameter(torch.zeros(HEAD))
        self.output_weight = nn.Parameter(torch.empty(classes, HEAD))
        self.output_bias = nn.Parameter(torch.zeros(classes))
        self.quantization_bits = 6
        self.qat = False
        self.storage_f16 = False
        self.reference_scan = False
        self.record_trace = False
        self.trace: dict[str, Tensor] = {}

        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.convolution, std=0.15)
        # Give different lanes short and long memories from the start.
        with torch.no_grad():
            self.gate_bias.copy_(torch.linspace(0.0, 4.0, HIDDEN))
        assert self.parameter_count() == self.expected_parameters(classes)

    @staticmethod
    def expected_parameters(classes: int = CLASSES) -> int:
        return (
            HIDDEN * FEATURES + HIDDEN
            + CONV_WIDTH * HIDDEN + HIDDEN
            + 2 * (HIDDEN * HIDDEN + HIDDEN)
            + HIDDEN * HIDDEN * 2 + HIDDEN
            + HEAD * HIDDEN * 2 + HEAD
            + classes * HEAD + classes
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def weight(self, name: str) -> Tensor:
        value = getattr(self, name)
        return quantize(value, self.quantization_bits) if self.qat else value

    def store(self, value: Tensor) -> Tensor:
        return half_storage(value) if self.storage_f16 else value

    def linear(self, value: Tensor, name: str) -> Tensor:
        return F.linear(value, self.weight(f"{name}_weight"), self.weight(f"{name}_bias"))

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        """features: (batch, length, 8) float; valid: (batch, length) bool -> (batch, classes)."""
        mask = valid.unsqueeze(-1)
        embedded = self.store(self.linear(features, "input")) * mask

        channels = embedded.transpose(1, 2)
        kernel = self.weight("convolution").transpose(0, 1).unsqueeze(1)
        encoded = F.conv1d(channels, kernel, padding=CONV_WIDTH // 2, groups=HIDDEN)
        encoded = encoded.transpose(1, 2) + self.weight("encoder_bias")
        encoded = self.store(torch.tanh(encoded)) * mask

        gate = self.store(torch.sigmoid(self.linear(encoded, "gate")))
        candidate = self.store((1 - gate) * torch.tanh(self.linear(encoded, "candidate")))
        gate = torch.where(mask, gate, torch.ones_like(gate))
        candidate = candidate * mask
        scan = sequential_scan if self.reference_scan else affine_scan
        forward = self.store(scan(gate, candidate))
        backward = self.store(scan(gate.flip(1), candidate.flip(1)).flip(1))
        combined = self.store(
            torch.tanh(encoded + self.linear(torch.cat((forward, backward), dim=-1), "combine"))
        )
        combined = combined * mask

        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mean = combined.sum(dim=1) / count
        # combined is in (-1, 1); padded positions sit below every real value.
        maximum = torch.where(mask, combined, torch.full_like(combined, -2.0)).amax(dim=1)
        pooled = self.store(torch.cat((mean, maximum), dim=-1))
        hidden = self.store(torch.tanh(self.linear(pooled, "head")))
        if self.record_trace:
            self.trace = {
                name: value.detach()
                for name, value in {
                    "embedded": embedded,
                    "encoded": encoded,
                    "gate": gate,
                    "candidate": candidate,
                    "forward": forward,
                    "backward": backward,
                    "combined": combined,
                    "pooled": pooled,
                    "hidden": hidden,
                }.items()
            }
        return self.linear(hidden, "output")
