#!/usr/bin/env python
"""Evaluate an OmniFold unfolding run: reweight the (held-out test) Pythia
sample with a trained step model and compare the resulting jet-substructure
distributions against raw Pythia and Herwig ("data"). Port of
ViniciusMikuni/OmniLearn's scripts/evaluate_omnifold.py -- same substructure
observables, same triangular-distance metric -- targeting this repo's PET2 /
PyTorch checkpoints (src/omnilearned/omnifold.py) instead of TF/Keras ones.

Deviates from the reference in one place: the reference's own
evaluate_omnifold.py evaluates against train_pythia.h5/train_herwig.h5; this
evaluates against the held-out test_pythia.h5/test_herwig.h5 instead (both
already exist alongside the train files under the --path directory), which
is the more standard choice and costs nothing extra since the test split was
already produced by preprocess_omnifold.py.

Usage (single process or `srun -n N` for faster inference -- ddp_setup()
degrades gracefully to a single-process group when no DDP env vars are set,
matching every other script in this project):

    python evaluate_omnifold.py --checkpoint-dir ... --save-tag <tag> \\
        --num-iter 5 [--reco] --path /pscratch/sd/t/twamorka/unfolding \\
        --plot-dir ./plots
"""

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from omnilearned.network import PET2
from omnilearned.omnifold import reweight
from omnilearned.utils import (
    ddp_setup,
    gather_tensors,
    get_checkpoint_name,
    get_model_parameters,
    is_master_node,
    restore_checkpoint,
)


def get_name_and_binning(nbins):
    binning = [
        np.linspace(0, 75, nbins),
        np.linspace(0, 0.6, nbins),
        np.linspace(0, 80, 80),
        np.linspace(-14, -2, nbins),
        np.linspace(0.0, 0.5, nbins),
        np.linspace(0.0, 1.2, nbins),
    ]
    name = [
        "Jet Mass [GeV]",
        "Jet Width",
        r"$n_{constituents}$",
        r"$ln\rho$",
        "$z_g$",
        r"$\tau_{21}$",
    ]
    return name, binning


def calculate_triangle_distance(feed_dict, weights, binning, alternative_name, reference_name, ntrials=100):
    w = np.abs(binning[1] - binning[0])
    x, _ = np.histogram(feed_dict[reference_name], weights=weights[reference_name], bins=binning)
    x2, _ = np.histogram(feed_dict[reference_name], weights=weights[reference_name] ** 2, bins=binning)
    x_norm = np.sum(x) * w
    y, _ = np.histogram(feed_dict[alternative_name], weights=weights[alternative_name], bins=binning)
    y2, _ = np.histogram(feed_dict[alternative_name], weights=weights[alternative_name] ** 2, bins=binning)
    y_norm = np.sum(y) * w

    dist = sum(
        0.5 * w * (x[ib] / x_norm - y[ib] / y_norm) ** 2 / (x[ib] / x_norm + y[ib] / y_norm)
        if x[ib] + y[ib] > 0
        else 0.0
        for ib in range(len(x))
    )

    x_plus, x_minus = x + np.sqrt(x2), x - np.sqrt(x2)
    y_plus, y_minus = y + np.sqrt(y2), y - np.sqrt(y2)
    rng = np.random.default_rng(0)
    results = []
    for _ in range(ntrials):
        x_ = rng.uniform(low=x_minus, high=x_plus)
        y_ = rng.uniform(low=y_minus, high=y_plus)
        d_ = sum(
            0.5 * w * (x_[ib] / x_norm - y_[ib] / y_norm) ** 2 / (x_[ib] / x_norm + y_[ib] / y_norm)
            if x_[ib] + y_[ib] > 0
            else 0.0
            for ib in range(len(x))
        )
        results.append(d_)
    return dist * 1e3, np.std(results) * 1e3


def plot_hist(feed_dict, weights, xlabel, binning, out_path, reference_name="herwig"):
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ref_hist, edges = np.histogram(
        feed_dict[reference_name], weights=weights[reference_name], bins=binning, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    for name, style in [
        ("pythia", {"histtype": "step", "color": "tab:blue"}),
        ("herwig", {"histtype": "step", "color": "black"}),
        ("pythia_unfolded", {"histtype": "step", "color": "tab:red"}),
    ]:
        if name not in feed_dict:
            continue
        hist, _ = np.histogram(
            feed_dict[name], weights=weights[name], bins=binning, density=True
        )
        ax.stairs(hist, edges, label=name, **style)
        if name != reference_name:
            ratio = np.divide(hist, ref_hist, out=np.zeros_like(hist), where=ref_hist > 0)
            ax_ratio.plot(centers, ratio, **{k: v for k, v in style.items() if k != "histtype"})
    ax.legend()
    ax.set_ylabel("Normalized")
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel(f"Ratio to {reference_name}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_model(checkpoint_dir, tag, num_feat, model_size, interaction, local_interaction, device):
    model_params = get_model_parameters(model_size)
    model = PET2(
        input_dim=num_feat,
        use_int=interaction,
        local_int=local_interaction,
        int_type="lhc",
        conditional=False,
        cond_dim=1,
        pid=False,
        pid_dim=1,
        add_info=False,
        add_dim=1,
        mode="classifier",
        num_classes=2,
        num_gen_classes=1,
        mlp_drop=0.0,
        attn_drop=0.0,
        feature_drop=0.0,
        num_coord=2,
        K=10,
        **model_params,
    )
    model.to(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    restore_checkpoint(model, checkpoint_dir, get_checkpoint_name(tag), device, is_main_node=is_master_node())
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--save-tag", required=True, help="Same --save-tag used at training time")
    parser.add_argument("--path", default="/pscratch/sd/t/twamorka/unfolding")
    parser.add_argument("--plot-dir", default="./plots")
    parser.add_argument("--num-iter", type=int, default=5, help="Iteration to load")
    parser.add_argument("--num-bins", type=int, default=50)
    parser.add_argument(
        "--reco", action="store_true",
        help="Load step-1 iteration-0 (event reweighting baseline) instead of the "
        "step-2 model at --num-iter",
    )
    parser.add_argument("--num-feat", type=int, default=13)
    parser.add_argument("--size", default="small")
    parser.add_argument("--interaction", action="store_true")
    parser.add_argument("--local-interaction", action="store_true")
    parser.add_argument("--batch", type=int, default=2048)
    args = parser.parse_args()

    local_rank, rank, size = ddp_setup()
    os.makedirs(args.plot_dir, exist_ok=True)

    with h5py.File(os.path.join(args.path, "test_pythia.h5"), "r") as f:
        n = f["reco"].shape[0]
        shard = np.arange(n)[rank::size]
        mc_reco = torch.from_numpy(f["reco"][:][shard].astype(np.float32))
        mc_gen = torch.from_numpy(f["gen"][:][shard].astype(np.float32))
        mc_reco_subs = f["reco_subs"][:][shard]
        mc_gen_subs = f["gen_subs"][:][shard]

    with h5py.File(os.path.join(args.path, "test_herwig.h5"), "r") as f:
        n = f["reco"].shape[0]
        shard = np.arange(n)[rank::size]
        data_reco_subs = f["reco_subs"][:][shard]

    step = 1 if args.reco else 2
    iteration = 0 if args.reco else args.num_iter - 1
    tag = f"{args.save_tag}_iter{iteration}_step{step}"
    if is_master_node():
        print(f"[evaluate_omnifold] loading {tag}")
    model = load_model(
        args.checkpoint_dir, tag, args.num_feat, args.size,
        args.interaction, args.local_interaction, local_rank,
    )

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if args.reco:
        weights = reweight(model, mc_reco, device, batch_size=args.batch)
        mc_subs = mc_reco_subs
        unfolded_name = "pythia_reweighted"
    else:
        weights = reweight(model, mc_gen, device, batch_size=args.batch)
        mc_subs = mc_gen_subs
        unfolded_name = "pythia_unfolded"

    weights_g = gather_tensors(weights)
    mc_subs_g = gather_tensors(torch.from_numpy(mc_subs))
    data_subs_g = gather_tensors(torch.from_numpy(data_reco_subs))

    if not is_master_node():
        return

    name, binning = get_name_and_binning(args.num_bins)
    for feat in range(mc_subs_g.shape[-1]):
        feed_dict = {
            "pythia": mc_subs_g[:, feat],
            unfolded_name: mc_subs_g[:, feat],
            "herwig": data_subs_g[:, feat],
        }
        w = {
            "pythia": np.ones(mc_subs_g.shape[0]),
            unfolded_name: weights_g.numpy(),
            "herwig": np.ones(data_subs_g.shape[0]),
        }
        out_path = os.path.join(args.plot_dir, f"omnifold_{tag}_feat{feat}.pdf")
        plot_hist(feed_dict, w, name[feat], binning[feat], out_path)

        d, derr = calculate_triangle_distance(
            feed_dict, w, binning[feat], alternative_name=unfolded_name, reference_name="herwig"
        )
        d_raw, derr_raw = calculate_triangle_distance(
            feed_dict, w, binning[feat], alternative_name="pythia", reference_name="herwig"
        )
        print(
            f"{name[feat]}: unfolded triangle-dist {d:.3f} +- {derr:.3f}  "
            f"(raw pythia: {d_raw:.3f} +- {derr_raw:.3f})"
        )


if __name__ == "__main__":
    main()
