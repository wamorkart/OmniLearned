"""Unit tests for the hls4ml-friendly DeepSets knobs.

``--act-layer relu`` and ``--deepsets-fixed-n N`` produce a body with no GELU,
no ``x[..., 2] != 0`` validity mask, and a plain mean pool over N fixed
leading-pT slots -- so the exported ONNX graph drops the Gelu / Equal / Not /
Cast / masked-ReduceSum ops that block hls4ml ingestion. The default
(``act_layer=nn.GELU``, ``fixed_n=0``) path must be untouched.
"""

import torch
import torch.nn as nn

from omnilearned.network import DeepSets, ACT_LAYERS

_DS_KW = dict(
    input_dim=4,
    num_classes=2,
    base_dim=32,
    num_phi_layers=2,
    num_rho_layers=1,
)


def _fake_batch(batch=6, n=150, n_real=40):
    x = torch.randn(batch, n, 4)
    x[:, n_real:, :] = 0.0
    x[:, :n_real, 2] = x[:, :n_real, 2].abs().sort(dim=1, descending=True).values
    y = torch.randint(0, 2, (batch,))
    return x, y


def test_default_is_unchanged():
    """fixed_n=0 + default act -> identical param count and state_dict keys."""
    base = DeepSets(**_DS_KW)
    also_base = DeepSets(**_DS_KW, fixed_n=0, act_layer=nn.GELU)
    assert sum(p.numel() for p in base.parameters()) == sum(
        p.numel() for p in also_base.parameters()
    )
    assert set(base.state_dict()) == set(also_base.state_dict())
    assert base.body.fixed_n == 0


def test_relu_swaps_activation():
    m = DeepSets(**_DS_KW, act_layer=ACT_LAYERS["relu"])
    kinds = {type(mod).__name__ for mod in m.modules()}
    assert "GELU" not in kinds
    assert "ReLU" in kinds
    # weight shapes are unchanged, so a GELU checkpoint still loads
    m.load_state_dict(DeepSets(**_DS_KW).state_dict(), strict=True)


def test_fixed_n_forward_backward():
    x, y = _fake_batch()
    m = DeepSets(**_DS_KW, act_layer=ACT_LAYERS["relu"], fixed_n=64)
    m.train()
    out = m(x, y)
    assert out["y_pred"].shape == (x.shape[0], 2)
    torch.nn.functional.cross_entropy(out["y_pred"], y).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


def test_fixed_n_truncation_matches_host_slice():
    """The body truncates to N leading-pT slots; feeding an already-sliced
    batch of exactly N must give the same output (host-side truncation is
    part of the model contract for export)."""
    x, y = _fake_batch(n_real=90)
    n = 64
    m = DeepSets(**_DS_KW, act_layer=ACT_LAYERS["relu"], fixed_n=n).eval()
    with torch.no_grad():
        a = m(x, y)["y_pred"]
        b = m(x[:, :n, :], y)["y_pred"]
    assert torch.allclose(a, b, atol=1e-6)


def test_fixed_n_ignores_validity_mask():
    """No in-graph mask: zeroing the 'padded' tail beyond n_real must still
    change the output (those rows are folded into the mean, not masked)."""
    n = 64
    m = DeepSets(**_DS_KW, act_layer=ACT_LAYERS["relu"], fixed_n=n).eval()
    x, y = _fake_batch(n_real=20)
    x2 = x.clone()
    x2[:, 30:n, :] = torch.randn_like(x2[:, 30:n, :])  # perturb "padded" slots
    with torch.no_grad():
        a = m(x, y)["y_pred"]
        b = m(x2, y)["y_pred"]
    assert not torch.allclose(a, b, atol=1e-4)


def test_fixed_n_dropped_when_interaction_on():
    """The message-passing path needs the mask, so fixed_n is ignored there."""
    m = DeepSets(**_DS_KW, fixed_n=64, num_interaction_layers=1, interaction_k=32)
    assert m.body.fixed_n == 0
