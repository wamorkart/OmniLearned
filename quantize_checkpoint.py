"""Post-training quantization (PTQ) step for the distill -> quantize -> fine-tune pipeline.

Takes a checkpoint produced by pretrain-level KD (e.g.
best_model_distill_pretrain_s_scratch_a05_b05_T4_full500_reg52.pt), round-trips
every float tensor in its body/classifier_head/generator_head state dicts
through a lower-precision dtype (default bfloat16) and back to float32, and
saves the result under a new tag. Storage stays float32 so the existing
restore_checkpoint()/fine-tune path (train.py, --fine-tune) can load it with
no changes -- the values themselves carry the quantization rounding error.

Usage:
    python quantize_checkpoint.py \
        --checkpoint-dir /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
        --tag distill_pretrain_s_scratch_a05_b05_T4_full500_reg52 \
        --dtype bfloat16
"""

import argparse
import os

import torch

STATE_DICT_KEYS = ["body", "classifier_head", "generator_head"]

DTYPE_CHOICES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def quantize_state_dict(state_dict, dtype):
    quantized = {}
    total_sq_err = 0.0
    total_sq_ref = 0.0
    max_abs_err = 0.0
    for key, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            quantized[key] = tensor
            continue
        rounded = tensor.to(dtype).to(torch.float32)
        err = (rounded - tensor).float()
        total_sq_err += err.pow(2).sum().item()
        total_sq_ref += tensor.float().pow(2).sum().item()
        max_abs_err = max(max_abs_err, err.abs().max().item() if err.numel() else 0.0)
        quantized[key] = rounded
    rel_l2 = (total_sq_err / total_sq_ref) ** 0.5 if total_sq_ref > 0 else 0.0
    return quantized, {"rel_l2": rel_l2, "max_abs_err": max_abs_err}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--tag", required=True, help="Source tag; reads best_model_<tag>.pt"
    )
    parser.add_argument(
        "--out-tag",
        default=None,
        help="Output tag; defaults to <tag>_<dtype-suffix>ptq",
    )
    parser.add_argument(
        "--dtype", choices=list(DTYPE_CHOICES), default="bfloat16"
    )
    args = parser.parse_args()

    dtype = DTYPE_CHOICES[args.dtype]
    dtype_suffix = {"bfloat16": "bf16", "float16": "fp16"}[args.dtype]
    out_tag = args.out_tag or f"{args.tag}_{dtype_suffix}ptq"

    src_path = os.path.join(args.checkpoint_dir, f"best_model_{args.tag}.pt")
    dst_path = os.path.join(args.checkpoint_dir, f"best_model_{out_tag}.pt")

    print(f"Loading {src_path}")
    checkpoint = torch.load(src_path, map_location="cpu")

    out_checkpoint = {}
    for key in STATE_DICT_KEYS:
        if key not in checkpoint:
            continue
        quantized, stats = quantize_state_dict(checkpoint[key], dtype)
        out_checkpoint[key] = quantized
        print(
            f"[{key}] quantized to {args.dtype}: "
            f"rel L2 error={stats['rel_l2']:.4e}, max abs error={stats['max_abs_err']:.4e}"
        )

    print(f"Saving {dst_path}")
    torch.save(out_checkpoint, dst_path)
    print(f"Done. Fine-tune with --pretrain-tag {out_tag} --fine-tune")


if __name__ == "__main__":
    main()
