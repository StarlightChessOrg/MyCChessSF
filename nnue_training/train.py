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
    apply_mate_label_remap,
    compute_zscore_stats,
    count_mate_labels,
    make_dataloader,
    pack_precomputed_dataset,
    parse_worker_count,
    precompute_features,
    resolve_train_workers,
    split_samples,
    split_samples_by_ratio,
)
from features.xqwl_psq import N_FEATURES
from labels import DEFAULT_MATE_THRESHOLD
from quiet import filter_static_samples, format_static_filter_stats


def log_train(message: str = "") -> None:
    """Print to stderr without breaking active tqdm bars."""
    import sys

    from tqdm import tqdm

    tqdm.write(message, file=sys.stderr)


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
        f"  val_mae_vl    {val_metrics['mae_vl']:.2f} (all)",
        f"  val_corr_vl   {val_metrics['corr_vl']:.4f} (all)",
        f"  val_mae_quiet {val_metrics['mae_vl_quiet']:.2f}",
        f"  val_corr_quiet {val_metrics['corr_vl_quiet']:.4f}  "
        f"(n={val_metrics['n_quiet']:,})",
    ]
    if val_metrics.get("n_mate", 0) > 0:
        lines.extend(
            [
                f"  val_mate      skipped n={val_metrics['n_mate']:,}",
                f"                (|vl|>={val_metrics['mate_threshold']:.0f})",
            ]
        )
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


def _masked_mse(preds, targets, mask):
    """MSE over samples where ``mask`` is True. Returns scalar loss."""
    import torch

    if not mask.any():
        return preds.sum() * 0.0
    diff = preds[mask] - targets[mask]
    return torch.mean(diff * diff)


def _accumulate_metrics(
    *,
    preds_norm,
    preds_vl,
    targets_norm,
    targets_raw,
    is_mate,
    criterion,
    totals: dict[str, float],
    counts: dict[str, int],
) -> None:
    import torch

    quiet = ~is_mate
    for key, mask in (("all", torch.ones_like(is_mate, dtype=torch.bool)), ("quiet", quiet)):
        if not mask.any():
            continue
        pn = preds_norm[mask]
        tn = targets_norm[mask]
        pv = preds_vl[mask]
        tr = targets_raw[mask]
        bs = int(mask.sum().item())

        totals[f"loss_{key}"] += criterion(pn, tn).item() * bs
        totals[f"abs_norm_{key}"] += (pn - tn).abs().sum().item()
        totals[f"sq_norm_{key}"] += ((pn - tn) ** 2).sum().item()
        totals[f"abs_vl_{key}"] += (pv - tr).abs().sum().item()

        totals[f"sum_pred_{key}"] += pv.sum().item()
        totals[f"sum_true_{key}"] += tr.sum().item()
        totals[f"sum_pred_sq_{key}"] += (pv * pv).sum().item()
        totals[f"sum_true_sq_{key}"] += (tr * tr).sum().item()
        totals[f"sum_cross_{key}"] += (pv * tr).sum().item()
        counts[key] += bs


def _finalize_metrics(
    totals: dict[str, float],
    counts: dict[str, int],
    *,
    mate_threshold: float,
    exclude_mate_from_loss: bool = False,
) -> dict[str, float]:
    def _corr(key: str) -> float:
        n = counts.get(key, 0)
        if n == 0:
            return math.nan
        mean_pred = totals[f"sum_pred_{key}"] / n
        mean_true = totals[f"sum_true_{key}"] / n
        var_pred = max(totals[f"sum_pred_sq_{key}"] / n - mean_pred * mean_pred, 0.0)
        var_true = max(totals[f"sum_true_sq_{key}"] / n - mean_true * mean_true, 0.0)
        cov = totals[f"sum_cross_{key}"] / n - mean_pred * mean_true
        denom = math.sqrt(var_pred * var_true)
        return cov / denom if denom > 1e-12 else 0.0

    n_all = counts.get("all", 0)
    n_quiet = counts.get("quiet", 0)
    n_mate = max(n_all - n_quiet, 0)
    if n_all == 0:
        return {
            "loss": math.inf,
            "mae_norm": math.inf,
            "rmse_norm": math.inf,
            "mae_vl": math.inf,
            "corr_vl": math.nan,
            "loss_quiet": math.inf,
            "mae_norm_quiet": math.inf,
            "rmse_norm_quiet": math.inf,
            "mae_vl_quiet": math.inf,
            "corr_vl_quiet": math.nan,
            "n_mate": n_mate,
            "n_quiet": n_quiet,
            "mate_threshold": mate_threshold,
        }

    use_quiet_loss = exclude_mate_from_loss and n_quiet > 0
    loss_key = "quiet" if use_quiet_loss else "all"
    n_loss = counts[loss_key]

    return {
        "loss": totals[f"loss_{loss_key}"] / n_loss,
        "mae_norm": totals[f"abs_norm_{loss_key}"] / n_loss,
        "rmse_norm": math.sqrt(totals[f"sq_norm_{loss_key}"] / n_loss),
        "mae_vl": totals["abs_vl_all"] / n_all,
        "corr_vl": _corr("all"),
        "loss_quiet": totals["loss_quiet"] / max(n_quiet, 1) if n_quiet else math.inf,
        "mae_norm_quiet": totals["abs_norm_quiet"] / max(n_quiet, 1) if n_quiet else math.inf,
        "rmse_norm_quiet": math.sqrt(totals["sq_norm_quiet"] / max(n_quiet, 1)) if n_quiet else math.inf,
        "mae_vl_quiet": totals["abs_vl_quiet"] / max(n_quiet, 1) if n_quiet else math.inf,
        "corr_vl_quiet": _corr("quiet") if n_quiet else math.nan,
        "n_mate": n_mate,
        "n_quiet": n_quiet,
        "mate_threshold": mate_threshold,
    }


def evaluate(
    model: Any,
    loader,
    device: Any,
    criterion,
    *,
    label_mean: float,
    label_std: float,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
    desc: str = "val",
    use_tqdm: bool = True,
    exclude_mate_from_loss: bool = False,
) -> dict[str, float]:
    import sys

    import torch
    from tqdm import tqdm

    model.eval()
    totals: dict[str, float] = {k: 0.0 for k in (
        "loss_all", "loss_quiet",
        "abs_norm_all", "abs_norm_quiet",
        "sq_norm_all", "sq_norm_quiet",
        "abs_vl_all", "abs_vl_quiet",
        "sum_pred_all", "sum_pred_quiet",
        "sum_true_all", "sum_true_quiet",
        "sum_pred_sq_all", "sum_pred_sq_quiet",
        "sum_true_sq_all", "sum_true_sq_quiet",
        "sum_cross_all", "sum_cross_quiet",
    )}
    counts = {"all": 0, "quiet": 0}

    non_blocking = device.type == "cuda"
    batch_iter = (
        tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
        if use_tqdm
        else loader
    )

    try:
        with torch.no_grad():
            for indices, offsets, targets_norm, targets_raw, is_mate in batch_iter:
                indices = indices.to(device, non_blocking=non_blocking)
                offsets = offsets.to(device, non_blocking=non_blocking)
                targets_norm = targets_norm.to(device, non_blocking=non_blocking)
                targets_raw = targets_raw.to(device, non_blocking=non_blocking)
                is_mate = is_mate.to(device, non_blocking=non_blocking)

                preds_norm = model(indices, offsets)
                preds_vl = preds_norm * label_std + label_mean
                _accumulate_metrics(
                    preds_norm=preds_norm,
                    preds_vl=preds_vl,
                    targets_norm=targets_norm,
                    targets_raw=targets_raw,
                    is_mate=is_mate,
                    criterion=criterion,
                    totals=totals,
                    counts=counts,
                )
    finally:
        if use_tqdm and isinstance(batch_iter, tqdm):
            batch_iter.close()

    return _finalize_metrics(
        totals,
        counts,
        mate_threshold=mate_threshold,
        exclude_mate_from_loss=exclude_mate_from_loss,
    )


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
    exclude_mate_from_loss: bool = False,
) -> float:
    import sys

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
            file=sys.stderr,
        )
        if use_tqdm
        else loader
    )

    try:
        for indices, offsets, targets_norm, _targets_raw, is_mate in batch_iter:
            indices = indices.to(device, non_blocking=non_blocking)
            offsets = offsets.to(device, non_blocking=non_blocking)
            targets_norm = targets_norm.to(device, non_blocking=non_blocking)
            is_mate = is_mate.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            preds = model(indices, offsets)
            if exclude_mate_from_loss:
                quiet = ~is_mate
                loss = _masked_mse(preds, targets_norm, quiet)
            else:
                loss = criterion(preds, targets_norm)
            loss.backward()
            optimizer.step()

            if exclude_mate_from_loss:
                bs = int((~is_mate).sum().item())
            else:
                bs = targets_norm.size(0)
            if bs > 0:
                total_loss += loss.item() * bs
                n += bs

            if use_tqdm and isinstance(batch_iter, tqdm):
                batch_iter.set_postfix(
                    loss=f"{total_loss / max(n, 1):.6f}",
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
    dedupe_fen = bool(data_cfg.get("dedupe_fen", True))
    _, load_workers_label = parse_worker_count(load_workers)
    print(
        f"[data] source={data_source}  pattern={pattern!r}  load_workers={load_workers_label}  "
        f"dedupe_fen={dedupe_fen}",
        flush=True,
    )

    if "val_ratio" in data_cfg:
        train_samples, val_samples, skipped = split_samples_by_ratio(
            data_source,
            pattern,
            float(data_cfg["val_ratio"]),
            seed=int(data_cfg.get("val_seed", 42)),
            load_workers=load_workers,
            dedupe_fen=dedupe_fen,
        )
    else:
        train_samples, val_samples, skipped = split_samples(
            data_source,
            pattern,
            data_cfg.get("val_workers", [30, 31]),
            load_workers=load_workers,
            dedupe_fen=dedupe_fen,
        )
    print(f"[data] train={len(train_samples):,}  val={len(val_samples):,}  skipped={skipped:,}", flush=True)

    mate_threshold = float(data_cfg.get("mate_threshold", DEFAULT_MATE_THRESHOLD))
    quiet_only = bool(data_cfg.get("quiet_only", True))
    quiet_pst_margin = float(data_cfg.get("quiet_pst_margin", 70.0))
    mate_remap = bool(data_cfg.get("mate_remap", not quiet_only))
    if quiet_only and bool(data_cfg.get("mate_remap", False)):
        print(
            "[data] quiet_only=true: ignoring mate_remap (mate-zone samples are dropped, not remapped)",
            flush=True,
        )
        mate_remap = False

    if quiet_only:
        train_samples, train_static = filter_static_samples(
            train_samples,
            mate_threshold=mate_threshold,
            pst_margin=quiet_pst_margin,
            load_workers=load_workers,
        )
        val_samples, val_static = filter_static_samples(
            val_samples,
            mate_threshold=mate_threshold,
            pst_margin=quiet_pst_margin,
            load_workers=load_workers,
        )
        print(f"[data] static filter train: {format_static_filter_stats(train_static)}", flush=True)
        print(f"[data] static filter val:   {format_static_filter_stats(val_static)}", flush=True)
        if not train_samples:
            raise ValueError("quiet_only removed all training samples; relax quiet_pst_margin or check data")
        print(
            f"[data] train={len(train_samples):,}  val={len(val_samples):,}  (after static filter)",
            flush=True,
        )
    else:
        print("[data] quiet_only=false: keeping all samples (in-check / mate / unstable PST)", flush=True)
    exclude_mate_from_zscore = bool(data_cfg.get("mate_exclude_from_zscore", False))
    exclude_mate_from_loss = bool(train_cfg.get("mate_exclude_from_loss", False))

    if mate_remap:
        train_mate_before, _ = count_mate_labels(train_samples, mate_threshold=mate_threshold)
        val_mate_before, _ = count_mate_labels(val_samples, mate_threshold=mate_threshold)
        train_samples, val_samples, remap_stats = apply_mate_label_remap(
            train_samples,
            val_samples,
            mate_threshold=mate_threshold,
        )
        print(
            f"[label] mate remap: quiet_min={remap_stats['quiet_min']:.1f}  "
            f"quiet_max={remap_stats['quiet_max']:.1f}  cap=±{remap_stats['mate_cap']:.1f}",
            flush=True,
        )
        print(
            f"[label]               remapped train={remap_stats['remapped_train']:,}/"
            f"{train_mate_before:,}  val={remap_stats['remapped_val']:,}/{val_mate_before:,}",
            flush=True,
        )
        exclude_mate_from_zscore = False
        exclude_mate_from_loss = False

    train_mate, train_quiet = count_mate_labels(train_samples, mate_threshold=mate_threshold)
    val_mate, val_quiet = count_mate_labels(val_samples, mate_threshold=mate_threshold)
    print(
        f"[label] mate zone |vl|>={mate_threshold:.0f}: "
        f"train {train_mate:,}/{len(train_samples):,}",
        flush=True,
    )
    print(
        f"[label]               val {val_mate:,}/{len(val_samples):,}",
        flush=True,
    )
    print(
        f"[label] mate handling: quiet_only={quiet_only}  remap={mate_remap}  "
        f"zscore_exclude={exclude_mate_from_zscore}  loss_exclude={exclude_mate_from_loss}",
        flush=True,
    )
    print("[label]               inference: pure NNUE when loaded, else pure PST", flush=True)

    print("[label] computing z-score stats ...", flush=True)
    label_mean, label_std = compute_zscore_stats(
        train_samples,
        exclude_mate=exclude_mate_from_zscore,
        mate_threshold=mate_threshold,
    )
    scope = "quiet-only" if exclude_mate_from_zscore else "all"
    print(f"[label] z-score mean={label_mean:.4f}  std={label_std:.4f}  ({scope})", flush=True)

    precompute = bool(train_cfg.get("precompute_features", use_cuda))
    feature_workers = train_cfg.get("feature_workers", data_cfg.get("load_workers", "auto"))
    if precompute:
        train_samples = precompute_features(train_samples, load_workers=feature_workers)
        val_samples = precompute_features(val_samples, load_workers=feature_workers)
        train_ds = pack_precomputed_dataset(
            train_samples,
            label_mean=label_mean,
            label_std=label_std,
            mate_threshold=mate_threshold,
        )
        val_ds = pack_precomputed_dataset(
            val_samples,
            label_mean=label_mean,
            label_std=label_std,
            mate_threshold=mate_threshold,
        )
        del train_samples, val_samples
    else:
        train_ds = NnueDataset(
            train_samples,
            label_mean=label_mean,
            label_std=label_std,
            mate_threshold=mate_threshold,
        )
        val_ds = NnueDataset(
            val_samples,
            label_mean=label_mean,
            label_std=label_std,
            mate_threshold=mate_threshold,
        )

    num_workers, num_workers_label = resolve_train_workers(
        train_cfg.get("num_workers", "auto"),
        use_cuda=use_cuda,
        precomputed=precompute,
    )
    batch_size = int(train_cfg["batch_size"])
    print(f"[loader] num_workers={num_workers_label}  batch_size={batch_size}", flush=True)
    print(
        f"[loader] prefetch={train_cfg.get('prefetch_factor', 2)}  "
        f"pin_memory={use_cuda}  storage={'csr' if precompute else 'samples'}",
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
    log_train(f"[model] params={param_count:,}")

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

    log_train(f"[train] max_epochs={max_epochs}  early_stopping_patience={patience}")
    if precompute and num_workers == 0:
        log_train(
            "[train] precomputed CSR dataset uses num_workers=0 to avoid copying GB of RAM "
            "into DataLoader worker processes (especially slow on Windows)."
        )
    log_train("")

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
            exclude_mate_from_loss=exclude_mate_from_loss,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            label_mean=label_mean,
            label_std=label_std,
            mate_threshold=mate_threshold,
            desc=f"val {epoch}/{max_epochs}",
            use_tqdm=use_tqdm,
            exclude_mate_from_loss=exclude_mate_from_loss,
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
