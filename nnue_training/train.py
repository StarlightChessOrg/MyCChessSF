"""Train MyCChessSF NNUE on nnue_data (XQWL-PSQ, z-score labels)."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    DEFAULT_DATA_PATTERN,
    NnueDataset,
    compute_zscore_stats,
    make_dataloader,
    pack_precomputed_dataset,
    parse_worker_count,
    precompute_features,
    resolve_train_workers,
    split_samples,
    split_samples_by_ratio,
)
from features.xqwl_psq import N_FEATURES


def log_train(message: str = "") -> None:
    """Print without breaking active tqdm bars."""
    from tqdm import tqdm

    tqdm.write(message)


def format_epoch_summary(
    *,
    epoch: int,
    max_epochs: int,
    elapsed_s: float,
    train_loss: float,
    val_metrics: dict[str, float],
    val_loss: float,
    best_val: float,
    is_best: bool,
    epochs_without_improvement: int,
    patience: int,
) -> str:
    lines = [
        "",
        f"── epoch {epoch}/{max_epochs} ({elapsed_s:.1f}s) ──",
        f"  train_loss    {train_loss:.6f}",
        f"  val_loss      {val_loss:.6f}" + ("  ← best" if is_best else ""),
        f"  val_mae_norm  {val_metrics['mae_norm']:.6f}",
        f"  val_rmse_norm {val_metrics['rmse_norm']:.6f}",
        f"  val_mae_vl    {val_metrics['mae_vl']:.2f}",
        f"  val_corr_vl   {val_metrics['corr_vl']:.4f}",
    ]
    if is_best:
        prev = "n/a" if not math.isfinite(best_val) else f"{best_val:.6f}"
        lines.append(f"  checkpoint    saved best.pt (prev best {prev})")
    else:
        lines.append(
            f"  early_stop    no improvement ({epochs_without_improvement}/{patience})"
        )
    return "\n".join(lines)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p).resolve()


def evaluate(
    model: Any,
    loader,
    device: Any,
    criterion,
    *,
    label_mean: float,
    label_std: float,
    desc: str = "val",
    use_tqdm: bool = True,
) -> dict[str, float]:
    import torch
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_abs_norm = 0.0
    total_sq_norm = 0.0
    total_abs_vl = 0.0
    n = 0

    sum_pred = 0.0
    sum_true = 0.0
    sum_pred_sq = 0.0
    sum_true_sq = 0.0
    sum_cross = 0.0

    non_blocking = device.type == "cuda"
    batch_iter = (
        tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        if use_tqdm
        else loader
    )

    try:
        with torch.no_grad():
            for indices, offsets, targets_norm, targets_raw in batch_iter:
                indices = indices.to(device, non_blocking=non_blocking)
                offsets = offsets.to(device, non_blocking=non_blocking)
                targets_norm = targets_norm.to(device, non_blocking=non_blocking)
                targets_raw = targets_raw.to(device, non_blocking=non_blocking)

                preds_norm = model(indices, offsets)
                loss = criterion(preds_norm, targets_norm)
                preds_vl = preds_norm * label_std + label_mean

                bs = targets_norm.size(0)
                total_loss += loss.item() * bs
                total_abs_norm += (preds_norm - targets_norm).abs().sum().item()
                total_sq_norm += ((preds_norm - targets_norm) ** 2).sum().item()
                total_abs_vl += (preds_vl - targets_raw).abs().sum().item()

                sum_pred += preds_vl.sum().item()
                sum_true += targets_raw.sum().item()
                sum_pred_sq += (preds_vl * preds_vl).sum().item()
                sum_true_sq += (targets_raw * targets_raw).sum().item()
                sum_cross += (preds_vl * targets_raw).sum().item()
                n += bs
    finally:
        if use_tqdm and isinstance(batch_iter, tqdm):
            batch_iter.close()

    if n == 0:
        return {
            "loss": math.inf,
            "mae_norm": math.inf,
            "rmse_norm": math.inf,
            "mae_vl": math.inf,
            "corr_vl": math.nan,
        }

    mean_pred = sum_pred / n
    mean_true = sum_true / n
    var_pred = max(sum_pred_sq / n - mean_pred * mean_pred, 0.0)
    var_true = max(sum_true_sq / n - mean_true * mean_true, 0.0)
    cov = sum_cross / n - mean_pred * mean_true
    denom = math.sqrt(var_pred * var_true)
    corr_vl = cov / denom if denom > 1e-12 else 0.0

    return {
        "loss": total_loss / n,
        "mae_norm": total_abs_norm / n,
        "rmse_norm": math.sqrt(total_sq_norm / n),
        "mae_vl": total_abs_vl / n,
        "corr_vl": corr_vl,
    }


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    epoch: int,
    val_loss: float,
    label_mean: float,
    label_std: float,
    config: dict,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "label_mean": label_mean,
            "label_std": label_std,
            "config": config,
            "n_features": N_FEATURES,
        },
        path,
    )


def train_epoch(
    model: Any,
    loader,
    device: Any,
    optimizer,
    criterion,
    *,
    epoch: int,
    max_epochs: int,
    use_tqdm: bool = True,
) -> float:
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    n = 0
    non_blocking = device.type == "cuda"
    batch_iter = (
        tqdm(
            loader,
            desc=f"train {epoch}/{max_epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        if use_tqdm
        else loader
    )

    try:
        for indices, offsets, targets_norm, _targets_raw in batch_iter:
            indices = indices.to(device, non_blocking=non_blocking)
            offsets = offsets.to(device, non_blocking=non_blocking)
            targets_norm = targets_norm.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            preds = model(indices, offsets)
            loss = criterion(preds, targets_norm)
            loss.backward()
            optimizer.step()

            bs = targets_norm.size(0)
            total_loss += loss.item() * bs
            n += bs

            if use_tqdm and isinstance(batch_iter, tqdm):
                batch_iter.set_postfix(
                    loss=f"{total_loss / n:.6f}",
                    samples=f"{n:,}",
                    refresh=False,
                )
    finally:
        if use_tqdm and isinstance(batch_iter, tqdm):
            batch_iter.close()

    return total_loss / max(n, 1)


def main() -> None:
    import torch
    import torch.nn as nn

    from model.nnue import NNUE

    parser = argparse.ArgumentParser(description="Train MyCChessSF NNUE")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    data_source = resolve_path(ROOT, data_cfg["source"])
    ckpt_dir = resolve_path(ROOT, train_cfg.get("checkpoint_dir", "checkpoints"))
    device = torch.device(train_cfg.get("device", "cpu"))
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print(f"[device] cuda={torch.cuda.get_device_name(device)}", flush=True)

    pattern = data_cfg.get("pattern", DEFAULT_DATA_PATTERN)
    load_workers = data_cfg.get("load_workers", "auto")
    _, load_workers_label = parse_worker_count(load_workers)
    print(
        f"[data] source={data_source}  pattern={pattern!r}  load_workers={load_workers_label}",
        flush=True,
    )

    if "val_ratio" in data_cfg:
        train_samples, val_samples, skipped = split_samples_by_ratio(
            data_source,
            pattern,
            float(data_cfg["val_ratio"]),
            seed=int(data_cfg.get("val_seed", 42)),
            load_workers=load_workers,
        )
    else:
        train_samples, val_samples, skipped = split_samples(
            data_source,
            pattern,
            data_cfg.get("val_workers", [30, 31]),
            load_workers=load_workers,
        )
    print(f"[data] train={len(train_samples):,}  val={len(val_samples):,}  skipped={skipped:,}", flush=True)

    print("[label] computing z-score stats ...", flush=True)
    label_mean, label_std = compute_zscore_stats(train_samples)
    print(f"[label] z-score mean={label_mean:.4f}  std={label_std:.4f}", flush=True)

    precompute = bool(train_cfg.get("precompute_features", use_cuda))
    feature_workers = train_cfg.get("feature_workers", data_cfg.get("load_workers", "auto"))
    if precompute:
        train_samples = precompute_features(train_samples, load_workers=feature_workers)
        val_samples = precompute_features(val_samples, load_workers=feature_workers)
        train_ds = pack_precomputed_dataset(
            train_samples,
            label_mean=label_mean,
            label_std=label_std,
        )
        val_ds = pack_precomputed_dataset(
            val_samples,
            label_mean=label_mean,
            label_std=label_std,
        )
        del train_samples, val_samples
    else:
        train_ds = NnueDataset(train_samples, label_mean=label_mean, label_std=label_std)
        val_ds = NnueDataset(val_samples, label_mean=label_mean, label_std=label_std)

    num_workers, num_workers_label = resolve_train_workers(
        train_cfg.get("num_workers", "auto"),
        use_cuda=use_cuda,
        precomputed=precompute,
    )
    batch_size = int(train_cfg["batch_size"])
    print(
        f"[loader] num_workers={num_workers_label}  batch_size={batch_size}  "
        f"prefetch={train_cfg.get('prefetch_factor', 2)}  pin_memory={use_cuda}  "
        f"storage={'csr' if precompute else 'samples'}",
        flush=True,
    )

    use_pin = use_cuda
    train_loader = make_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
        pin_memory=use_pin,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
        pin_memory=use_pin,
    )

    model = NNUE(
        n_features=model_cfg.get("n_features", N_FEATURES),
        l1=model_cfg["l1"],
        l2=model_cfg["l2"],
        l3=model_cfg["l3"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[model] params={param_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    criterion = nn.MSELoss()

    best_val = math.inf
    max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 384)))
    patience = int(train_cfg.get("early_stopping_patience", 32))
    use_tqdm = bool(train_cfg.get("tqdm", True))
    epochs_without_improvement = 0

    print(f"[train] max_epochs={max_epochs}  early_stopping_patience={patience}", flush=True)
    if precompute and num_workers == 0:
        print(
            "[train] precomputed CSR dataset uses num_workers=0 to avoid copying GB of RAM "
            "into DataLoader worker processes (especially slow on Windows).",
            flush=True,
        )

    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            epoch=epoch,
            max_epochs=max_epochs,
            use_tqdm=use_tqdm,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            label_mean=label_mean,
            label_std=label_std,
            desc=f"val {epoch}/{max_epochs}",
            use_tqdm=use_tqdm,
        )
        val_loss = val_metrics["loss"]
        epoch_elapsed = time.time() - epoch_t0
        prev_best = best_val
        is_best = val_loss < prev_best

        save_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            label_mean=label_mean,
            label_std=label_std,
            config=cfg,
        )

        if is_best:
            best_val = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                label_mean=label_mean,
                label_std=label_std,
                config=cfg,
            )
        else:
            epochs_without_improvement += 1

        if use_tqdm:
            log_train(
                format_epoch_summary(
                    epoch=epoch,
                    max_epochs=max_epochs,
                    elapsed_s=epoch_elapsed,
                    train_loss=train_loss,
                    val_metrics=val_metrics,
                    val_loss=val_loss,
                    best_val=prev_best,
                    is_best=is_best,
                    epochs_without_improvement=epochs_without_improvement,
                    patience=patience,
                )
            )
        else:
            print(
                f"epoch {epoch}/{max_epochs} done ({epoch_elapsed:.1f}s)  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"val_mae_norm={val_metrics['mae_norm']:.6f}  "
                f"val_rmse_norm={val_metrics['rmse_norm']:.6f}  "
                f"val_mae_vl={val_metrics['mae_vl']:.2f}  val_corr_vl={val_metrics['corr_vl']:.4f}",
                flush=True,
            )
            if is_best:
                print(f"  saved best.pt  val_loss={val_loss:.6f}", flush=True)
            else:
                print(
                    f"  no val_loss improvement  "
                    f"({epochs_without_improvement}/{patience})",
                    flush=True,
                )

        if not is_best and epochs_without_improvement >= patience:
            log_train(
                f"[early stop] patience={patience} reached at epoch {epoch}  "
                f"best val_loss={best_val:.6f}"
            )
            break

    log_train(f"[done] best val_loss={best_val:.6f}")
    log_train(f"checkpoints: {ckpt_dir / 'best.pt'} , {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    if multiprocessing.current_process().name == "MainProcess":
        main()
