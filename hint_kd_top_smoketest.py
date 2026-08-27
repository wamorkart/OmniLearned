"""Cheap feasibility smoke test for hint/feature-based KD on top tagging:
student = distill_top_small_scratch_a00_b10_T4 (PET2-small, our best/winning
KD checkpoint), teacher = fine_tune_top_l (PET2-large).

Unlike the DeepSets case, PET2's forward() already returns a real x_body
(the body's token-sequence output) for both architectures -- no network.py
change needed. Reuses get_distill_cls_loss from utils.py, the same MSE-after-
projector utility train.py's --distill-cls path uses in production.

Does NOT precompute teacher embeddings across the dataset (that's the real
cost of a production hint-KD pipeline, via build_teacher_h5.py
--include-cls-embed). Runs the teacher LIVE for a short number of
iterations, just to check the mechanism works and see an early loss trend.

Standalone -- does not edit network.py/train.py/cli.py/utils.py.
Must be run through a real GPU allocation (salloc/sbatch), not the login
node -- the teacher's --interaction attention bias is memory-heavy.

Usage:
    /global/homes/t/twamorka/omnilearned-clean/env/bin/python hint_kd_top_smoketest.py
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from omnilearned.dataloader import load_data
from omnilearned.network import PET2
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
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
BATCH = 32  # kept small: teacher is 373.7M params w/ --interaction (quadratic attn bias)
DISTILL_T = 4.0
DISTILL_ALPHA, DISTILL_BETA = 0.0, 1.0  # matches the a00_b10 winning recipe
HINT_GAMMA = 0.5  # matches train.py's --distill-gamma default


def main():
    local_rank, rank, size = ddp_setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Teacher: PET2 large, --interaction (matches fine_tune_top_l's own
    # training config in train.sh), frozen, eval mode.
    teacher_params = get_model_parameters("large")
    teacher = PET2(input_dim=4, use_int=True, mode="classifier", num_classes=2, **teacher_params)
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

    # Student: PET2 small, --interaction --local-interaction (matches
    # distill_train_top.sh, the script family behind this checkpoint),
    # warm-started from the winning pure-KD checkpoint.
    student_params = get_model_parameters("small")
    student = PET2(
        input_dim=4, use_int=True, local_int=True, mode="classifier", num_classes=2, **student_params
    )
    restore_checkpoint(
        student, CHECKPOINT_DIR, get_checkpoint_name("distill_top_small_scratch_a00_b10_T4"),
        local_rank, is_main_node=True,
    )
    student.to(device).train()
    student_num_tokens = student_params["num_tokens"]
    student_dim = student_params["base_dim"]
    print(f"Student loaded: distill_top_small_scratch_a00_b10_T4 (small), "
          f"{sum(p.numel() for p in student.parameters()):,} params, "
          f"num_tokens={student_num_tokens}, base_dim={student_dim}")

    # Same convention as train.py's --distill-cls path: flatten the first
    # num_tokens CLS-style tokens, project student's flat dim -> teacher's.
    student_flat_dim = student_num_tokens * student_dim
    teacher_flat_dim = teacher_num_tokens * teacher_dim
    projector = nn.Linear(student_flat_dim, teacher_flat_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()), lr=1e-6
    )

    train_loader = load_data(
        "top", dataset_type="train", use_cond=False, path=DATA_PATH, batch=BATCH,
        num_workers=2, rank=rank, size=size, mode="classifier",
        teacher_labels_dir=TEACHER_DIR, teacher_tag="fine_tune_top_l",
    )
    data_iter = iter(train_loader)

    logs = {"loss_class": [], "loss_kd": [], "loss_hint": [], "loss_total": []}

    for it in tqdm(range(NUM_ITERS), desc="hint-KD (top, PET2) smoke test"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        X = batch["X"].to(device, dtype=torch.float)
        y = batch["y"].to(device)
        teacher_logits = batch["teacher_logits"].to(device)

        optimizer.zero_grad()

        outputs = student(X, y)
        loss_class = get_loss(
            outputs, y, "classifier", nn.CrossEntropyLoss(), nn.MSELoss(),
            False, False, None, {"loss_class": torch.zeros(1, device=device),
                                   "loss_gen": torch.zeros(1, device=device),
                                   "loss_clip": torch.zeros(1, device=device),
                                   "loss_class_event": torch.zeros(1, device=device),
                                   "loss": torch.zeros(1, device=device)},
        )
        loss_kd = get_distill_loss(outputs["y_pred"], teacher_logits, distill_T=DISTILL_T)

        with torch.no_grad():
            teacher_out = teacher(X, y)
            teacher_embed = teacher_out["x_body"][:, :teacher_num_tokens].reshape(X.shape[0], -1)

        student_embed = outputs["x_body"][:, :student_num_tokens].reshape(X.shape[0], -1)
        loss_hint = get_distill_cls_loss(student_embed, teacher_embed, projector)

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
