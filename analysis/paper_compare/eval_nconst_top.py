"""Comparison #4 -- robustness to a reduced number of input constituents,
mirroring Petitjean et al. arXiv:2512.17011 Fig. 6 (right): they restrict the
input to the 30 highest-pT constituents (a top jet peaks around 50) and every
tagger loses a significant amount of background rejection.

Here we re-evaluate an EXISTING top-tagging checkpoint (PET2 or DeepSets) on
the test split with the input truncated per jet to the N leading-pT
constituents -- no retraining. Feature index 2 is pT (see
dataloader.py __getitem__: `sample["X"][:, 2] > 0.0` is the validity test),
so we argsort by it descending and keep the first N.

  python analysis/paper_compare/eval_nconst_top.py \
      --tag distill_top_small_scratch_a05_T4_r1 --arch pet2 --size small --nconst 30
  # --nconst 0  -> no truncation (full-N reference)

Appends one row to <outdir>/nconst_results.tsv.
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from omnilearned.dataloader import load_data
from omnilearned.network import PET2, DeepSets
from omnilearned.utils import (
    ddp_setup, get_checkpoint_name, get_model_parameters,
    get_deepsets_parameters, restore_checkpoint,
)

CHECKPOINT_DIR = "/pscratch/sd/t/twamorka/omnilearned/checkpoints/"
DATA_PATH = "/global/cfs/cdirs/m4567/www/"
SIGNAL_EFFS = [0.50, 0.30]
PT_FEAT = 2


def build_model(arch, size):
    if arch == "pet2":
        p = get_model_parameters(size)
        return PET2(input_dim=4, use_int=True, local_int=True, num_classes=2,
                    mode="classifier", num_coord=2, K=10, **p)
    p = get_deepsets_parameters(size)
    return DeepSets(input_dim=4, num_classes=2, mode="classifier", **p)


def truncate_to_leading_pt(X, nconst):
    """X: (B, M, F). Keep the nconst entries with the largest feature-2 (pT)
    per row; zero-padding rows sort to the end so real leading constituents
    are kept."""
    if nconst <= 0 or nconst >= X.shape[1]:
        return X
    order = torch.argsort(X[:, :, PT_FEAT], dim=1, descending=True)
    Xs = torch.gather(X, 1, order.unsqueeze(-1).expand(-1, -1, X.shape[2]))
    return Xs[:, :nconst, :].contiguous()


def compute_metrics(preds, labels):
    scores = preds[:, 1]
    acc = (preds.argmax(1) == labels).mean()
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    rej = {}
    for eff in SIGNAL_EFFS:
        idx = tpr.searchsorted(eff)
        rej[eff] = float("inf") if idx >= len(fpr) or fpr[idx] == 0 else 1.0 / fpr[idx]
    return acc, auc, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", required=True, choices=["pet2", "deepsets"])
    ap.add_argument("--size", required=True)
    ap.add_argument("--nconst", type=int, required=True, help="0 => no truncation")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--outdir", default="/pscratch/sd/t/twamorka/omnilearned/results/paper_compare")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    local_rank, rank, size = ddp_setup()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = build_model(args.arch, args.size)
    restore_checkpoint(model, CHECKPOINT_DIR, get_checkpoint_name(args.tag),
                       local_rank, is_main_node=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.tag} ({args.arch}/{args.size}): {n_params:,} params  nconst={args.nconst}")

    loader = load_data("top", dataset_type="test", use_cond=True, path=DATA_PATH,
                       batch=args.batch, num_workers=args.num_workers, rank=0, size=1,
                       mode="classifier", shuffle=False)

    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"nconst={args.nconst}", total=len(loader)):
            X = batch["X"].to(device, dtype=torch.float)
            y = batch["y"].to(device)
            cond = batch["cond"].to(device) if batch.get("cond") is not None else None
            X = truncate_to_leading_pt(X, args.nconst)
            out = model(X, y, cond=cond)
            preds.append(out["y_pred"].float().softmax(-1).cpu())
            labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    acc, auc, rej = compute_metrics(preds, labels)
    print(f"\n=== {args.tag}  nconst={args.nconst} ===")
    print(f"  N events : {len(labels):,}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  1/FPR@50%: {rej[0.5]:.1f}")
    print(f"  1/FPR@30%: {rej[0.3]:.1f}")

    row = f"{args.tag}\t{args.arch}\t{args.size}\t{n_params}\t{args.nconst}\t{acc:.4f}\t{auc:.4f}\t{rej[0.5]:.2f}\t{rej[0.3]:.2f}\n"
    path = os.path.join(args.outdir, "nconst_results.tsv")
    header = "tag\tarch\tsize\tparams\tnconst\tacc\tauc\trej50\trej30\n"
    if not os.path.exists(path):
        open(path, "w").write(header)
    open(path, "a").write(row)
    print("appended ->", path)


if __name__ == "__main__":
    main()
