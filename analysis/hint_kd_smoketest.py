"""Cheap feasibility smoke test for hint/feature-based KD on the DeepSets
"small" student, distilled from the fine_tune_top_l teacher.

Does NOT precompute teacher embeddings across the dataset (that's the real
cost of a production hint-KD pipeline, via build_teacher_h5.py
--include-cls-embed). Instead runs the teacher LIVE (frozen, no grad) for a
short number of iterations, just to check the mechanism works and see an
early loss trend before committing to building the full pipeline.

Adds one small trainable Linear projector (student's pooled body embedding z
-> teacher's CLS-token dim), reusing get_distill_cls_loss (MSE) from
utils.py. Warm-starts the student from its existing best KD checkpoint
(distill_top_deepsets_small_scratch_a05_T4_archfix0804), NOT from scratch.

Standalone -- does not edit network.py/train.py/cli.py/utils.py.

Usage:
    /global/homes/t/twamorka/omnilearned-clean/env/bin/python hint_kd_smoketest.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from omnilearned.dataloader import load_data
from omnilearned.network import PET2, DeepSets
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
    get_deepsets_parameters,
    get_distill_cls_loss,
    get_distill_loss,
    get_loss,
    get_model_parameters,
    is_master_node,
    restore_checkpoint,
)

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
TEACHER_DIR = "/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l"
NUM_ITERS = 100
BATCH = 16
DISTILL_ALPHA, DISTILL_BETA, DISTILL_T = 0.5, 0.5, 4.0
HINT_GAMMA = 0.1


def main():
    local_rank, rank, size = ddp_setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Teacher: PET2 large, frozen, eval mode.
    teacher_params = get_model_parameters("large")
    teacher = PET2(
        input_dim=4, use_int=True, mode="classifier", num_classes=2, **teacher_params
    )
    restore_checkpoint(
        teacher, CHECKPOINT_DIR, get_checkpoint_name("fine_tune_top_l"), local_rank, is_main_node=True
    )
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_num_tokens = teacher_params["num_tokens"]
    teacher_dim = teacher_params["base_dim"]
    print(f"Teacher loaded: fine_tune_top_l, {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params, "
          f"num_tokens={teacher_num_tokens}, base_dim={teacher_dim}")

    # Student: DeepSets small, warm-started from its best KD checkpoint.
    ds_params = get_deepsets_parameters("small")
    student = DeepSets(input_dim=4, num_classes=2, mode="classifier", **ds_params)
    restore_checkpoint(
        student, CHECKPOINT_DIR, get_checkpoint_name("distill_top_deepsets_small_scratch_a05_T4_archfix0804"),
        local_rank, is_main_node=True,
    )
    student.to(device).train()
    student_dim = ds_params["base_dim"]
    print(f"Student loaded: distill_top_deepsets_small_scratch_a05_T4_archfix0804 (small), "
          f"{sum(p.numel() for p in student.parameters()):,} params, base_dim={student_dim}")

    projector = nn.Linear(student_dim, teacher_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()), lr=5e-5
    )

    train_loader = load_data(
        "top", dataset_type="train", use_cond=True, path=DATA_PATH, batch=BATCH,
        num_workers=2, rank=rank, size=size, mode="classifier",
        teacher_labels_dir=TEACHER_DIR, teacher_tag="fine_tune_top_l",
    )
    data_iter = iter(train_loader)

    logs = {"loss_class": [], "loss_kd": [], "loss_hint": [], "loss_total": []}

    for it in tqdm(range(NUM_ITERS), desc="hint-KD smoke test"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        X = batch["X"].to(device, dtype=torch.float)
        y = batch["y"].to(device)
        cond = batch["cond"].to(device) if batch.get("cond") is not None else None
        teacher_logits = batch["teacher_logits"].to(device)

        optimizer.zero_grad()

        z = student.body(X, cond=cond)
        y_pred = student.classifier(z)
        outputs = {"y_pred": y_pred, "z_pred": None, "y_perturb": None}
        loss_logs = {
            "loss": torch.zeros(1, device=device),
            "loss_class": torch.zeros(1, device=device),
            "loss_gen": torch.zeros(1, device=device),
            "loss_clip": torch.zeros(1, device=device),
            "loss_class_event": torch.zeros(1, device=device),
        }
        loss_class = get_loss(
            outputs, y, "classifier", nn.CrossEntropyLoss(), nn.MSELoss(),
            False, False, None, loss_logs,
        )
        loss_kd = get_distill_loss(y_pred, teacher_logits, distill_T=DISTILL_T)

        with torch.no_grad():
            teacher_out = teacher(X, y, cond=cond)
            teacher_cls = teacher_out["x_body"][:, :teacher_num_tokens].mean(dim=1)

        loss_hint = get_distill_cls_loss(z, teacher_cls, projector)

        loss = DISTILL_ALPHA * loss_class + DISTILL_BETA * loss_kd + HINT_GAMMA * loss_hint
        loss.backward()
        optimizer.step()

        logs["loss_class"].append(loss_class.item())
        logs["loss_kd"].append(loss_kd.item())
        logs["loss_hint"].append(loss_hint.item())
        logs["loss_total"].append(loss.item())

        if it % 10 == 0 or it == NUM_ITERS - 1:
            print(f"iter {it:3d}  class={loss_class.item():.4f}  kd={loss_kd.item():.4f}  "
                  f"hint={loss_hint.item():.4f}  total={loss.item():.4f}")

    print("\n=== Summary (first 5 vs last 5 iters) ===")
    for key in logs:
        first5 = sum(logs[key][:5]) / 5
        last5 = sum(logs[key][-5:]) / 5
        print(f"{key:12s}  first5={first5:.4f}  last5={last5:.4f}  "
              f"{'DOWN' if last5 < first5 else 'UP'}")


if __name__ == "__main__":
    main()
