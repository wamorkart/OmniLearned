import torch
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_checkpoint_name,
    restore_checkpoint,
    pad_array,
    get_model_parameters,
)
from omnilearned.diffusion import generate
import os
import time
import numpy as np
import h5py
from tqdm.auto import tqdm
from torch.utils.flop_counter import FlopCounterMode


def eval_model(
    model,
    test_loader,
    dataset,
    mode,
    use_event_loss,
    device="cpu",
    outdir="",
    save_tag="pretrain",
    rank=0,
    dataset_type="test",
    num_chunks=1,
    chunk_idx=0,
):
    chunk_suffix = f"_chunk{chunk_idx}of{num_chunks}" if num_chunks > 1 else ""
    cls_name = f"outputs_{save_tag}_{dataset}_{dataset_type}{chunk_suffix}_rank{rank}.npz"
    gen_name = f"generated_{save_tag}_{dataset}_{dataset_type}{chunk_suffix}_rank{rank}.h5"
    # For classifier mode we also collect sample_keys + raw logits so the
    # same outputs_*.npz file can serve as the teacher labels for KD.
    return_sample_keys = mode == "classifier"
    result = test_step(
        model, test_loader, mode, device, return_sample_keys=return_sample_keys
    )
    if return_sample_keys:
        prediction, cond, labels, sample_keys = result
    else:
        prediction, cond, labels = result

    if mode in ["classifier", "regression", "segmentation"]:
        if use_event_loss:
            np.savez(
                os.path.join(
                    outdir, cls_name
                ),
                prediction=prediction[:, :200].softmax(-1).cpu().numpy(),
                event_prediction=prediction[:, 200:].softmax(-1).cpu().numpy(),
                logits=prediction.cpu().numpy().astype(np.float16),
                sample_keys=sample_keys.cpu().numpy(),
                pid=labels.cpu().numpy(),
                cond=cond.cpu().numpy() if cond is not None else [],
            )
        else:
            if mode == "classifier":
                logits_np = prediction.cpu().numpy().astype(np.float16)
                prediction_np = prediction.softmax(-1).cpu().numpy()
                np.savez(
                    os.path.join(
                        outdir,
                        cls_name,
                    ),
                    prediction=prediction_np,
                    logits=logits_np,
                    sample_keys=sample_keys.cpu().numpy(),
                    pid=labels.cpu().numpy(),
                    cond=cond.cpu().numpy() if cond is not None else [],
                )
            else:
                np.savez(
                    os.path.join(
                        outdir,
                        cls_name,
                    ),
                    prediction=prediction.cpu().numpy(),
                    pid=labels.cpu().numpy(),
                    cond=cond.cpu().numpy() if cond is not None else [],
                )
    else:
        with h5py.File(
            os.path.join(
                outdir, gen_name
            ),
            "w",
        ) as fh5:
            fh5.create_dataset("data", data=prediction.cpu().numpy())
            fh5.create_dataset("global", data=cond.cpu().numpy())
            fh5.create_dataset("pid", data=labels.cpu().numpy() + 1)


def test_step(
    model,
    dataloader,
    mode,
    device,
    return_sample_keys=False,
):
    model.eval()

    # Inference precision is controlled by EVAL_AMP: "fp32" (default, no
    # autocast), "bf16", or "fp16". Logits are saved as float16 regardless, so
    # bf16 autocast gives a large speedup on tensor cores at negligible cost --
    # but the default stays fp32 until the bf16 path is validated A/B, so a
    # run can't silently produce unvalidated bf16 logits. Set EVAL_AMP=bf16 to
    # opt in per-dataset.
    amp_choice = os.environ.get("EVAL_AMP", "fp32").lower()
    amp_dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    on_cuda = device != "cpu"
    use_amp = on_cuda and amp_choice in amp_dtypes
    amp_dtype = amp_dtypes.get(amp_choice)
    if is_master_node():
        print(f"[eval] forward precision: {'autocast ' + amp_choice if use_amp else 'fp32'}")
    fwd_time = 0.0

    # Match input dtype to the model's own weight dtype: stays fp32 for
    # unquantized/int8/int4 models (torchao's quantized tensor subclasses
    # report fp32 as their outer dtype), and becomes fp16 when QUANTIZE=fp16
    # has cast the model's weights with .half().
    model_dtype = next(model.parameters()).dtype

    preds = []
    labels = []
    conds = []
    sample_keys = [] if return_sample_keys else None

    for ib, batch in enumerate(
        tqdm(dataloader, desc="Iterating", total=len(dataloader))
        if is_master_node()
        else dataloader
    ):
        X, y = batch["X"].to(device, dtype=model_dtype), batch["y"].to(device)
        npart = X.shape[1]
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }

        with torch.no_grad():
            if mode in ["classifier", "regression", "segmentation"]:
                if on_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if use_amp:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        outputs = model(X, y, **model_kwargs)
                else:
                    outputs = model(X, y, **model_kwargs)
                if on_cuda:
                    torch.cuda.synchronize()
                fwd_time += time.perf_counter() - t0
                output_name = (
                    "y_pred" if mode in ["classifier", "regression"] else "z_pred"
                )
                # Cast back to fp32: autocast outputs bf16/fp16, but downstream
                # softmax + numpy save need a numpy-representable float dtype.
                preds.append(outputs[output_name].float())

            elif mode == "generator":
                assert "cond" in model_kwargs, (
                    "ERROR, conditioning variables not passed to model"
                )
                preds.append(generate(model, y, X.shape, **model_kwargs))
        if mode == "segmentation":
            labels.append(batch["data_pid"].to(device))
        else:
            labels.append(y)

        conds.append(batch["cond"])
        if return_sample_keys:
            sample_keys.append(batch["sample_key"])
        if mode == "generator":
            if batch["pid"] is not None:
                preds[-1] = torch.cat(
                    [preds[-1], model_kwargs["pid"].unsqueeze(-1).float()], -1
                )
            if batch["add_info"] is not None:
                preds[-1] = torch.cat([preds[-1], model_kwargs["add_info"]], -1)

    if is_master_node() and mode in ["classifier", "regression", "segmentation"]:
        print(f"[eval] total forward time: {fwd_time:.2f}s over {len(preds)} batches "
              f"({1000 * fwd_time / max(len(preds), 1):.1f} ms/batch)")

    if mode == "generator":
        preds = pad_array(preds, npart)
    else:
        preds = torch.cat(preds).to(device)

    result = (
        preds,
        torch.cat(conds).to(device) if conds[0] is not None else None,
        torch.cat(labels).to(device),
    )
    if return_sample_keys:
        result = result + (torch.cat(sample_keys),)
    return result


def run(
    indir: str = "",
    outdir: str = "",
    save_tag: str = "",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    num_feat: int = 4,
    model_size: str = "small",
    interaction: bool = False,
    local_interaction: bool = False,
    num_coord: int = 2,
    K: int = 10,
    interaction_type: str = "lhc",
    conditional: bool = False,
    num_cond: int = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    use_add: bool = False,
    num_add: int = 4,
    use_event_loss: bool = False,
    num_classes: int = 2,
    num_gen_classes: int = 1,
    mode: str = "classifier",
    batch: int = 64,
    num_workers: int = 16,
    clip_inputs: bool = False,
    dataset_type: str = "test",
    num_chunks: int = 1,
    chunk_idx: int = 0,
):
    local_rank, rank, size = ddp_setup()

    model_params = get_model_parameters(model_size)

    # set up model
    model = PET2(
        input_dim=num_feat,
        use_int=interaction,
        local_int=local_interaction,
        int_type=interaction_type,
        conditional=conditional,
        cond_dim=num_cond,
        pid=use_pid,
        add_info=use_add,
        add_dim=num_add,
        mode=mode,
        num_classes=num_classes,
        num_gen_classes=num_gen_classes,
        num_coord=num_coord,
        K=K,
        **model_params,
    )

    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Evaluating on device: {d}, with {size} GPUs")
        print("************")

    test_loader = load_data(
        dataset,
        dataset_type=dataset_type,
        use_cond=True,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        clip_inputs=clip_inputs,
        mode=mode,
        shuffle=False,
        num_chunks=num_chunks,
        chunk_idx=chunk_idx,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(test_loader)}")
        print("************")

    if os.path.isfile(os.path.join(indir, get_checkpoint_name(save_tag))):
        if is_master_node():
            print(
                f"Loading checkpoint from {os.path.join(indir, get_checkpoint_name(save_tag))}"
            )

        restore_checkpoint(
            model,
            indir,
            get_checkpoint_name(save_tag),
            local_rank,
            is_main_node=is_master_node(),
            restore_ema_model=mode == "generator",
        )

    else:
        raise ValueError(
            f"Error loading checkpoint: {os.path.join(indir, get_checkpoint_name(save_tag))}"
        )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    if is_master_node():
        # FLOPs are a property of the architecture (number of multiply-adds),
        # not of the numeric precision used to run them, so this is measured
        # once here on the plain model -- before quantization/DDP -- rather
        # than repeated per QUANTIZE mode.
        sample_batch = next(iter(test_loader))
        sample_X = sample_batch["X"].to(device, dtype=torch.float)
        sample_y = sample_batch["y"].to(device)
        sample_kwargs = {
            key: (sample_batch[key].to(device) if sample_batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in sample_batch
        }
        with torch.no_grad(), FlopCounterMode(display=False) as flop_counter:
            model(sample_X, sample_y, **sample_kwargs)
        print("**** Setup ****")
        print(
            "FLOPs per forward pass (batch=%d): %.3f GFLOPs"
            % (sample_X.shape[0], flop_counter.get_total_flops() / 1e9)
        )
        print("************")

    model = DDP(
        model,
        **kwarg,
    )

    # --- Weight precision reduction (optional) ---
    # fp16 and int4 are disabled for now, pending a decision on the
    # architecture change they'd require (InteractionBlock and the attention
    # masks hardcode float32 in layers.py/network.py, which breaks a hard
    # .half() cast and torchao's packed int4 kernel). Uncomment below to
    # re-enable once that's resolved.
    quantize_choices = ("none", "int8")  # "fp16", "int4" temporarily disabled
    quantize_choice = os.environ.get("QUANTIZE", "none").lower()
    if quantize_choice not in quantize_choices:
        raise ValueError(f"QUANTIZE must be one of {quantize_choices}, got '{quantize_choice}'")

    # if quantize_choice == "fp16":
    #     # Weights are NOT cast/stored as fp16 here -- InteractionBlock and the
    #     # attention masks hardcode float32 (layers.py/network.py), so a hard
    #     # .half() on the whole model crashes with a dtype mismatch. Instead
    #     # this reuses EVAL_AMP's autocast path: PyTorch runs the big matmuls
    #     # in fp16 and automatically keeps numerically fragile ops (log, exp,
    #     # softmax) in fp32. That's a compute-precision speedup, not a weight
    #     # compression -- the model stays fp32-sized in memory, unlike
    #     # int8/int4 below.
    #     os.environ.setdefault("EVAL_AMP", "fp16")
    #     if is_master_node():
    #         print("[eval] QUANTIZE=fp16 requests fp16 autocast for the forward pass")
    if quantize_choice == "int8":
        from torchao.quantization import quantize_, int8_weight_only
        if is_master_node():
            print("[eval] applying INT8 weight-only quantization")
        quantize_(model.module, int8_weight_only())
    # elif quantize_choice == "int4":
    #     if device == "cpu":
    #         raise RuntimeError("INT4 weight-only quantization requires a CUDA device")
    #     from torchao.quantization import quantize_, int4_weight_only
    #
    #     def _int4_filter(module, fqn):
    #         # torchao's int4 tensor-core kernel requires in_features to be a
    #         # multiple of group_size. The handful of raw-feature input
    #         # projections (in_features 3/4/7) don't meet that and are a
    #         # negligible share of total parameters, so leave them unquantized
    #         # instead of erroring out.
    #         group_size = 128
    #         return isinstance(module, torch.nn.Linear) and module.in_features % group_size == 0
    #
    #     if is_master_node():
    #         print("[eval] applying INT4 weight-only quantization")
    #     quantize_(model.module, int4_weight_only(group_size=128), filter_fn=_int4_filter)

    eval_model(
        model,
        test_loader,
        dataset,
        mode=mode,
        use_event_loss=use_event_loss,
        device=device,
        rank=rank,
        outdir=outdir,
        save_tag=save_tag,
        dataset_type=dataset_type,
        num_chunks=num_chunks,
        chunk_idx=chunk_idx,
    )
    dist.barrier()
    dist.destroy_process_group()
