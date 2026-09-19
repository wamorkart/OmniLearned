"""OmniFold iterative unfolding, ported from ViniciusMikuni/OmniLearn's
omnifold.py / train_omnifold.py (TensorFlow/Horovod) to this repo's PET2 /
PyTorch / DDP stack.

Algorithm (Andreassen et al., standard 2-step iterative OmniFold): treat
Pythia as MC (has both reco and gen truth), Herwig as a stand-in for "data"
(reco only). Each iteration:
  Step 1 (reco reweighting): classify MC-reco vs Data-reco, reweight MC-reco
    events by the classifier's likelihood ratio -> weights_pull.
  Step 2 (gen-to-gen push): classify MC-gen (weight=1) vs the SAME MC-gen
    events (weight=weights_pull), pushing the reco-level reweighting down to
    gen level via the reco/gen correlation -> weights_push (normalized) for
    the next iteration.
Both step classifiers are persistent across iterations -- only the training
targets/weights change each iteration, weights are never reset.

Deliberately reuses train.py::train_model() as the inner "RunModel" step
(AMP/EMA/DDP/early-stopping/wandb/checkpointing all come for free) instead of
reimplementing a training loop. The one piece of new plumbing this needed --
threading an externally supplied per-event weight through get_loss/get_class_loss
instead of the internal class-balance weighting -- lives in utils.py/train.py.

Deliberately does NOT use dataloader.py's HEPDataset/load_data (built for
lazy, sharded, billion-sample HDF5 mixtures). This dataset is ~1.6M+1.6M
events -- small enough to load fully into memory once per process.
"""

import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from pytorch_optimizer import Lion
from diffusers.optimization import get_cosine_schedule_with_warmup

from omnilearned.network import PET2
from omnilearned.train import train_model
from omnilearned.dataloader import collate_point_cloud
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
    get_last_checkpoint_name,
    get_model_parameters,
    get_param_groups,
    is_master_node,
    restore_checkpoint,
    save_checkpoint,
    shadow_copy,
)

VAL_FRACTION = 0.1


class OmniFoldArrayDataset(Dataset):
    """Wraps a fixed (X, y, weight) triple already sliced to this rank's shard.

    Takes the place of HEPDataset for this task -- rebuilt (cheap: just
    re-wraps existing tensors with new labels/weights) at the start of every
    Step1/Step2 call in a new iteration.
    """

    def __init__(self, X, y, weight):
        self.X = X
        self.y = y
        self.weight = weight

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "y": self.y[idx],
            "weight": self.weight[idx],
            # collate_point_cloud requires sample_key unconditionally; unused
            # downstream for this task.
            "sample_key": torch.tensor([idx, 0], dtype=torch.int64),
        }


def _make_loader(X, y, weight, batch_size, num_workers):
    dataset = OmniFoldArrayDataset(X, y, weight)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_point_cloud,
        drop_last=False,
    )


def _rank_shard(n, rank, size):
    """Same rank::size stride sharding dataloader.py's load_data uses for the
    lazy HDF5 file_indices array -- kept here so every rank trains on a
    disjoint subset with plain shuffle=True (no DistributedSampler needed)."""
    return np.arange(n)[rank::size]


def _train_val_split(n, rng):
    idx = rng.permutation(n)
    n_val = max(1, int(n * VAL_FRACTION))
    return idx[n_val:], idx[:n_val]


class OmniFoldSample:
    """One sample (mc or data): loads reco/gen particle+jet arrays for a
    single h5 file fully into memory, shards by rank, and holds a 90/10
    train/val split fixed for the lifetime of the run."""

    def __init__(self, path, rank, size, has_gen, seed=0):
        with h5py.File(path, "r") as f:
            self.reco = torch.from_numpy(np.asarray(f["reco"], dtype=np.float32))
            self.gen = (
                torch.from_numpy(np.asarray(f["gen"], dtype=np.float32))
                if has_gen
                else None
            )
        n_total = self.reco.shape[0]
        shard = _rank_shard(n_total, rank, size)
        self.reco = self.reco[shard]
        if self.gen is not None:
            self.gen = self.gen[shard]
        self.nevts = self.reco.shape[0]
        self.weight = torch.ones(self.nevts, dtype=torch.float32)

        rng = np.random.default_rng(seed + rank)
        self.train_idx, self.val_idx = _train_val_split(self.nevts, rng)


def reweight(model, X, device, batch_size=2048):
    """Batched inference -> likelihood ratio p(class=1)/p(class=0) from the
    2-class softmax output. PyTorch-native analogue of the TF reference's
    sigmoid(logit)/(1-sigmoid(logit)) trick -- this repo's classifier mode is
    already 2-class softmax/CE everywhere (matches --dataset dctr's
    --num-classes 2 convention) rather than a single-sigmoid-logit head."""
    model.eval()
    weights = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = X[start : start + batch_size].to(device, dtype=torch.float)
            outputs = model(batch, None)
            probs = torch.softmax(outputs["y_pred"], dim=-1)
            f = probs[:, 1].clamp(1e-7, 1 - 1e-7)
            w = f / (1.0 - f)
            weights.append(w.cpu())
    model.train()
    return torch.cat(weights, dim=0)


def _normalize(weight, device):
    """Rescale weight so its (global, all-rank) sum matches its (global)
    count -- ports omnifold.py's norm_factor logic (hvd.allreduce -> divide)
    to plain dist.all_reduce, one SUM per rank's local total."""
    local_sum = torch.tensor([weight.sum().item()], dtype=torch.float64, device=device)
    local_count = torch.tensor(
        [float(weight.numel())], dtype=torch.float64, device=device
    )
    if dist.is_initialized():
        dist.all_reduce(local_sum)
        dist.all_reduce(local_count)
    norm_factor = (local_sum / local_count).item()
    return weight / norm_factor


class OmniFold:
    def __init__(
        self,
        save_tag,
        outdir,
        pretrain_tag,
        fine_tune,
        num_iter,
        patience,
        model_kwargs,
        batch,
        epoch,
        warmup_epoch,
        lr,
        lr_factor,
        wd,
        b1,
        b2,
        optim,
        sched,
        use_amp,
        amp_dtype,
        num_workers,
        wandb_flag,
        local_rank,
        rank,
        size,
    ):
        self.save_tag = save_tag
        self.outdir = outdir
        self.pretrain_tag = pretrain_tag
        self.fine_tune = fine_tune
        self.num_iter = num_iter
        self.patience = patience
        self.model_kwargs = model_kwargs
        self.batch = batch
        self.epoch = epoch
        self.warmup_epoch = warmup_epoch
        self.lr = lr
        self.lr_factor = lr_factor
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.optim = optim
        self.sched = sched
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.num_workers = num_workers
        self.wandb_flag = wandb_flag
        self.local_rank = local_rank
        self.rank = rank
        self.size = size
        self.device = local_rank if torch.cuda.is_available() else "cpu"

        self.model1 = self._new_model()  # Step 1: reco reweighting
        self.model2 = self._new_model()  # Step 2: gen-to-gen push
        self.loss_class = nn.CrossEntropyLoss(reduction="none")
        self.loss_gen = nn.MSELoss(reduction="none")
        self.run_wandb = None

    def _new_model(self):
        model_params = get_model_parameters(self.model_kwargs["model_size"])
        model = PET2(
            input_dim=self.model_kwargs["num_feat"],
            use_int=self.model_kwargs["interaction"],
            local_int=self.model_kwargs["local_interaction"],
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
        if torch.cuda.is_available():
            model.to(self.local_rank)
            model = DDP(model, device_ids=[self.local_rank])
        else:
            model.cpu()
            model = DDP(model)
        return model

    def _tag(self, iteration, step):
        return f"{self.save_tag}_iter{iteration}_step{step}"

    def _step_done(self, iteration, step):
        return os.path.isfile(
            os.path.join(self.outdir, f"training_{self._tag(iteration, step)}.json")
        )

    def _load_lineage(self, model, iteration, step):
        """Loads the correct starting weights for (iteration, step)'s model:
        - iteration 0: pretrain checkpoint if --fine-tune, else fresh init.
        - iteration i>0: this same step's own best checkpoint from iteration i-1
          (the persistent-model-across-iterations semantics of OmniFold)."""
        ema_model = shadow_copy(model.module)
        if iteration == 0:
            if self.fine_tune:
                restore_checkpoint(
                    model,
                    self.outdir,
                    get_checkpoint_name(self.pretrain_tag),
                    self.local_rank,
                    is_main_node=is_master_node(),
                    ema_model=ema_model,
                    fine_tune=True,
                )
            return ema_model
        prev_tag = self._tag(iteration - 1, step)
        restore_checkpoint(
            model,
            self.outdir,
            get_checkpoint_name(prev_tag),
            self.local_rank,
            is_main_node=is_master_node(),
            ema_model=ema_model,
            fine_tune=False,
        )
        return ema_model

    def _run_model(self, model, iteration, step, X_tr, y_tr, w_tr, X_val, y_val, w_val):
        tag = self._tag(iteration, step)
        if self._step_done(iteration, step):
            if is_master_node():
                print(f"[omnifold] {tag} already complete, loading best checkpoint")
            restore_checkpoint(
                model,
                self.outdir,
                get_checkpoint_name(tag),
                self.local_rank,
                is_main_node=is_master_node(),
            )
            return

        ema_model = self._load_lineage(model, iteration, step)

        base_model = model.module
        param_groups = get_param_groups(
            base_model, self.wd, self.lr, lr_factor=self.lr_factor, fine_tune=False
        )
        optimizer = (
            Lion(param_groups, betas=(self.b1, self.b2))
            if self.optim == "lion"
            else torch.optim.AdamW(param_groups)
        )

        train_loader = _make_loader(X_tr, y_tr, w_tr, self.batch, self.num_workers)
        val_loader = _make_loader(X_val, y_val, w_val, self.batch, self.num_workers)
        train_steps = len(train_loader)

        if self.sched == "onecycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                total_steps=train_steps * self.epoch,
                max_lr=self.lr,
                pct_start=0.1,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=train_steps * self.warmup_epoch,
                num_training_steps=train_steps * self.epoch,
            )

        epoch_init, loss_init, best_epoch_init = 0, np.inf, None
        last_ckpt = os.path.join(self.outdir, get_last_checkpoint_name(tag))
        if os.path.isfile(last_ckpt):
            if is_master_node():
                print(f"[omnifold] resuming {tag} from last checkpoint")
            epoch_init, loss_init, best_epoch_init = restore_checkpoint(
                model,
                self.outdir,
                get_last_checkpoint_name(tag),
                self.local_rank,
                is_main_node=is_master_node(),
                ema_model=ema_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                fine_tune=False,
            )

        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            lr_scheduler,
            mode="classifier",
            num_epochs=self.epoch,
            device=self.device,
            patience=self.patience,
            loss_class=self.loss_class,
            loss_gen=self.loss_gen,
            output_dir=self.outdir,
            save_tag=tag,
            epoch_init=epoch_init,
            loss_init=loss_init,
            best_epoch_init=best_epoch_init,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            run=self.run_wandb,
            ema_model=ema_model,
        )
        # train_model always leaves the best-checkpoint weights on disk, not
        # necessarily on `model` in memory (the last epoch trained may not be
        # the best one) -- reload so the next iteration's lineage carries the
        # actual best state forward, matching what evaluation will later load.
        restore_checkpoint(
            model,
            self.outdir,
            get_checkpoint_name(tag),
            self.local_rank,
            is_main_node=is_master_node(),
        )

    def run_step1(self, iteration, mc, data, weights_push):
        """Data vs MC reco reweighting -> weights_pull (mc-local shard)."""
        X = torch.cat([mc.reco, data.reco], dim=0)
        y = torch.cat(
            [torch.zeros(mc.nevts, dtype=torch.int64), torch.ones(data.nevts, dtype=torch.int64)]
        )
        w = torch.cat([weights_push * mc.weight, data.weight])

        mc_tr, mc_val = mc.train_idx, mc.val_idx
        data_tr, data_val = data.train_idx, data.val_idx
        tr_idx = np.concatenate([mc_tr, mc.nevts + data_tr])
        val_idx = np.concatenate([mc_val, mc.nevts + data_val])

        self._run_model(
            self.model1,
            iteration,
            1,
            X[tr_idx],
            y[tr_idx],
            w[tr_idx],
            X[val_idx],
            y[val_idx],
            w[val_idx],
        )

        new_weights = reweight(self.model1, mc.reco, self.device)
        weights_pull = weights_push * new_weights
        return _normalize(weights_pull, self.device)

    def run_step2(self, iteration, mc, weights_pull):
        """Gen-to-gen push: same mc.gen events, twice, different weights ->
        weights_push (mc-local shard)."""
        X = torch.cat([mc.gen, mc.gen], dim=0)
        y = torch.cat(
            [torch.zeros(mc.nevts, dtype=torch.int64), torch.ones(mc.nevts, dtype=torch.int64)]
        )
        w = torch.cat([mc.weight, mc.weight * weights_pull])

        mc_tr, mc_val = mc.train_idx, mc.val_idx
        tr_idx = np.concatenate([mc_tr, mc.nevts + mc_tr])
        val_idx = np.concatenate([mc_val, mc.nevts + mc_val])

        self._run_model(
            self.model2,
            iteration,
            2,
            X[tr_idx],
            y[tr_idx],
            w[tr_idx],
            X[val_idx],
            y[val_idx],
            w[val_idx],
        )

        new_weights = reweight(self.model2, mc.gen, self.device)
        return _normalize(new_weights, self.device)

    def unfold(self, mc, data):
        weights_push = torch.ones(mc.nevts, dtype=torch.float32)
        for i in range(self.num_iter):
            if is_master_node():
                print(f"[omnifold] ITERATION {i + 1}/{self.num_iter}")
            weights_pull = self.run_step1(i, mc, data, weights_push)
            weights_push = self.run_step2(i, mc, weights_pull)


def run(
    outdir="",
    save_tag="",
    pretrain_tag="",
    path="/pscratch/sd/t/twamorka/unfolding",
    wandb=False,
    fine_tune=False,
    num_feat=13,
    model_size="small",
    interaction=False,
    local_interaction=False,
    num_iter=5,
    patience=3,
    batch=512,
    epoch=30,
    warmup_epoch=1,
    lr=3e-5,
    lr_factor=5.0,
    wd=0.1,
    b1=0.95,
    b2=0.99,
    optim="lion",
    sched="cosine",
    use_amp=False,
    amp_dtype="fp16",
    num_workers=0,
):
    amp_dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if amp_dtype not in amp_dtypes:
        raise ValueError(f"--amp-dtype must be one of {list(amp_dtypes)}, got '{amp_dtype}'")

    local_rank, rank, size = ddp_setup()

    mc = OmniFoldSample(
        os.path.join(path, "train_pythia.h5"), rank, size, has_gen=True, seed=0
    )
    data = OmniFoldSample(
        os.path.join(path, "train_herwig.h5"), rank, size, has_gen=False, seed=1000
    )
    if is_master_node():
        print(f"[omnifold] mc (pythia) shard: {mc.nevts} events")
        print(f"[omnifold] data (herwig) shard: {data.nevts} events")

    ofold = OmniFold(
        save_tag=save_tag,
        outdir=outdir,
        pretrain_tag=pretrain_tag,
        fine_tune=fine_tune,
        num_iter=num_iter,
        patience=patience,
        model_kwargs={
            "num_feat": num_feat,
            "model_size": model_size,
            "interaction": interaction,
            "local_interaction": local_interaction,
        },
        batch=batch,
        epoch=epoch,
        warmup_epoch=warmup_epoch,
        lr=lr,
        lr_factor=lr_factor,
        wd=wd,
        b1=b1,
        b2=b2,
        optim=optim,
        sched=sched,
        use_amp=use_amp,
        amp_dtype=amp_dtypes[amp_dtype],
        num_workers=num_workers,
        wandb_flag=wandb,
        local_rank=local_rank,
        rank=rank,
        size=size,
    )

    if wandb:
        import wandb as wandb_module

        mode_wandb = None if is_master_node() else "disabled"
        if is_master_node():
            wandb_module.login()
        ofold.run_wandb = wandb_module.init(
            project="OmniBoone",
            name=save_tag,
            mode=mode_wandb,
            config={
                "learning_rate": lr,
                "epochs": epoch,
                "batch_size": batch,
                "num_iter": num_iter,
                "fine_tune": fine_tune,
            },
        )

    ofold.unfold(mc, data)
