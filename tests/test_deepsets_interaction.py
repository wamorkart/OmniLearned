"""Unit tests for the optional DeepSets GNN (message-passing) layers.

The interaction stack is opt-in: ``num_interaction_layers == 0`` must be
bit-for-bit the original Deep Sets body, and existing checkpoints must keep
loading with ``strict=True``.
"""

import torch

from omnilearned.network import DeepSets

_DS_KW = dict(
    input_dim=4,
    num_classes=2,
    base_dim=32,
    num_phi_layers=2,
    num_rho_layers=1,
)


def _fake_batch(batch=6, n=150, n_real=40):
    """A padded, pT-descending point cloud (feature index 2 is log pT)."""
    x = torch.randn(batch, n, 4)
    x[:, n_real:, :] = 0.0
    x[:, :n_real, 2] = x[:, :n_real, 2].abs().sort(dim=1, descending=True).values
    y = torch.randint(0, 2, (batch,))
    return x, y


def test_default_is_unchanged():
    """No interaction layers -> no extra params, identical state_dict keys."""
    base = DeepSets(**_DS_KW)
    with_k_only = DeepSets(**_DS_KW, interaction_k=64)  # ignored when layers == 0

    assert sum(p.numel() for p in base.parameters()) == sum(
        p.numel() for p in with_k_only.parameters()
    )
    assert set(base.state_dict()) == set(with_k_only.state_dict())
    # a plain checkpoint still loads strictly
    DeepSets(**_DS_KW).load_state_dict(base.state_dict(), strict=True)


def test_interaction_adds_params_and_keys():
    m = DeepSets(**_DS_KW, num_interaction_layers=2)
    extra = set(m.state_dict()) - set(DeepSets(**_DS_KW).state_dict())
    assert extra, "interaction layers added no parameters"
    assert all(k.startswith("body.interaction") for k in extra)


def test_forward_backward():
    x, y = _fake_batch()
    m = DeepSets(**_DS_KW, mlp_drop=0.1, num_interaction_layers=2, interaction_k=32)
    m.train()
    out = m(x, y)
    assert out["y_pred"].shape == (x.shape[0], 2)
    loss = torch.nn.functional.cross_entropy(out["y_pred"], y)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
    )


def test_truncation_matches_manual_slice():
    """interaction_k truncation is a plain leading-pT slice, so feeding an
    already-sliced batch must give the same output."""
    x, y = _fake_batch(n_real=80)
    k = 32
    m = DeepSets(**_DS_KW, num_interaction_layers=1, interaction_k=k).eval()

    m_nok = DeepSets(**_DS_KW, num_interaction_layers=1, interaction_k=0).eval()
    m_nok.load_state_dict(m.state_dict())

    with torch.no_grad():
        a = m(x, y)["y_pred"]
        b = m_nok(x[:, :k, :], y)["y_pred"]
    assert torch.allclose(a, b, atol=1e-5)
