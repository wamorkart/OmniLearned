"""Regression tests for checkpoint resume bookkeeping in
``omnilearned.utils.restore_checkpoint``.
"""

import torch
import torch.nn as nn

from omnilearned.utils import restore_checkpoint


class _TinyModel(nn.Module):
    """Minimal stand-in for PET2: restore_checkpoint only touches
    ``.body`` / ``.classifier`` / ``.generator``."""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.classifier = None
        self.generator = None


def _write_checkpoint(path, epoch, loss=0.5):
    torch.save(
        {
            "body": _TinyModel().body.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def test_resume_does_not_skip_an_epoch(tmp_path):
    """``save_checkpoint`` stores ``epoch + 1`` (epochs completed), so the
    resume point must be exactly that value -- the training loop is
    ``for epoch in range(epoch_init, num_epochs)``. A previous ``+ 1`` here
    silently dropped one epoch of training on every resume."""
    completed_epochs = 7
    ckpt = tmp_path / "best_model_unit.pt"
    _write_checkpoint(ckpt, epoch=completed_epochs)

    start_epoch, _best_loss = restore_checkpoint(
        _TinyModel(), str(tmp_path), ckpt.name, device="cpu"
    )

    assert start_epoch == completed_epochs


def test_resume_preserves_best_loss(tmp_path):
    ckpt = tmp_path / "best_model_unit.pt"
    _write_checkpoint(ckpt, epoch=3, loss=0.1234)

    _start_epoch, best_loss = restore_checkpoint(
        _TinyModel(), str(tmp_path), ckpt.name, device="cpu"
    )

    assert best_loss == 0.1234
