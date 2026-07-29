"""Pure-PyTorch stand-ins for the flash-attn ops used outside the attention kernel.

flash-attn's prebuilt wheels do not ship the `fused_dense_lib` CUDA extension, so `FusedDense`
and `FusedMLP` are unavailable unless it is compiled from source, and `layer_norm_fn` is triton
and therefore CUDA-only. These reproduce them exactly, so state dicts and outputs are unchanged.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

RMSNorm = nn.RMSNorm


class FusedDense(nn.Linear):
    """flash-attn's FusedDense also subclasses nn.Linear, so the parameters are identical."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        return_residual: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.return_residual = return_residual

    def forward(self, x, process_group=None):
        out = F.linear(x, self.weight, self.bias)
        return out if not self.return_residual else (out, x)


class FusedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias1: bool = True,
        bias2: bool = True,
        activation: str = "gelu_approx",
        return_residual: bool = False,
        checkpoint_lvl: int = 0,
        heuristic: str = "auto",
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.activation = activation
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x, process_group=None):
        pre_act = self.fc1(x)
        if self.activation == "gelu_approx":
            hidden = F.gelu(pre_act, approximate="tanh")
        elif self.activation == "relu":
            hidden = F.relu(pre_act)
        elif self.activation == "sqrelu":
            hidden = F.relu(pre_act) ** 2
        else:
            raise ValueError(f"unknown activation {self.activation}")
        out = self.fc2(hidden)
        return out if not self.return_residual else (out, x)


def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    zero_centered_weight=False,
    is_rms_norm=False,
    return_dropout_mask=False,
    out_dtype=None,
    out=None,
    residual_out=None,
):
    assert x1 is None and not return_dropout_mask and out is None and residual_out is None
    dtype = x.dtype
    if zero_centered_weight:
        weight = weight + 1.0
    if rowscale is not None:
        x = x * rowscale[..., None]
    if dropout_p > 0.0:
        x = F.dropout(x, p=dropout_p)

    # The triton kernel accumulates the residual in fp32 and normalizes that accumulator, so a
    # naive `x + residual` in the input dtype would drift from it.
    residual_dtype = residual.dtype if residual is not None else (
        torch.float32 if residual_in_fp32 else None
    )
    acc = x if residual_dtype is None else x.to(residual_dtype)
    if residual is not None:
        acc = acc + residual

    if is_rms_norm:
        normed = acc * torch.rsqrt(acc.float().pow(2).mean(-1, keepdim=True) + eps).to(acc.dtype)
        normed = normed * weight + (bias if bias is not None else 0.0)
    else:
        normed = F.layer_norm(acc, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    normed = normed.to(out_dtype or dtype)
    return normed if not prenorm else (normed, acc)
