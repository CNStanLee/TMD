"""
run_all.py  —  TMD 4-bit Brevitas 1D-CNN
=========================================
Preprocess  →  Train  →  Evaluate

Dataset format (train.dat / test.dat):
    {{x,y,z},label},
    e.g. {{-0.6946377,12.680544,0.50395286},0},

Classes: 0=Still  1=Walking  2=Run  3=Bike  4=Car/Bus/Train

Features
--------
* Sliding-window segmentation with configurable window / stride
* Per-channel z-score normalisation (stats from train split only)
* Gaussian jitter augmentation on training windows
* Class-frequency-weighted cross-entropy (handles imbalance)
* AdamW + linear LR warm-up + cosine annealing
* AMP (mixed-precision) on CUDA
* Gradient clipping
* Train / validation split (stratified, 10 % of train set)
* Early stopping on validation loss
* tqdm progress bars per epoch
* Per-class precision / recall / F1 + confusion matrix at the end
* Best and last checkpoint saving
"""

import os
import re
import sys
import time
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from models import TMD_1DCNN, export_model_to_qonnx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WINDOW_SIZE   = 128
STRIDE_TRAIN  = 64
STRIDE_TEST   = 128
BATCH_SIZE    = 256
NUM_EPOCHS    = 60
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
VAL_SPLIT     = 0.10
PATIENCE      = 10
JITTER_STD    = 0.05
GRAD_CLIP     = 1.0
NUM_WORKERS   = min(4, os.cpu_count() or 1)
SEED          = 42
SCRIPT_DIR    = Path(__file__).parent
DATA_DIR      = SCRIPT_DIR / "dataset"
OUT_DIR       = SCRIPT_DIR / "runs" / time.strftime("%Y%m%d_%H%M%S")
CLASS_NAMES   = ["Still", "Walking", "Run", "Bike", "Vehicle"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tmd")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# 1. Data parsing
# ---------------------------------------------------------------------------
def parse_dat(filepath: Path, log: logging.Logger) -> tuple:
    """Parse {{x,y,z},label} lines → float32 (N,3) array + int64 (N,) labels."""
    pattern = re.compile(r'\{\{([^}]+)\},\s*(\d+)\}')
    xs, ys, zs, labels = [], [], [], []
    with open(filepath) as f:
        for line in f:
            m = pattern.search(line)
            if m is None:
                continue
            x, y, z = (float(v) for v in m.group(1).split(","))
            xs.append(x); ys.append(y); zs.append(z)
            labels.append(int(m.group(2)))
    data   = np.column_stack([xs, ys, zs]).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    log.info("  parsed %s  →  %d samples", filepath.name, len(labels))
    return data, labels


# ---------------------------------------------------------------------------
# 2. Sliding-window segmentation
# ---------------------------------------------------------------------------
def sliding_windows(data: np.ndarray, labels: np.ndarray,
                    window: int, stride: int) -> tuple:
    """
    Returns
    -------
    X  (N, 3, window)  float32 – channel-first
    y  (N,)            int64   – majority-vote label per window
    """
    n      = len(data)
    starts = list(range(0, n - window + 1, stride))
    X = np.empty((len(starts), 3, window), dtype=np.float32)
    y = np.empty(len(starts), dtype=np.int64)
    for i, s in enumerate(starts):
        X[i] = data[s:s + window].T
        y[i] = int(np.bincount(labels[s:s + window]).argmax())
    return X, y


# ---------------------------------------------------------------------------
# 3. Normalisation
# ---------------------------------------------------------------------------
def compute_stats(X: np.ndarray) -> tuple:
    """Per-channel mean / std from training windows. X shape: (N, 3, T)."""
    flat = X.reshape(X.shape[0], X.shape[1], -1)
    mean = flat.mean(axis=(0, 2))
    std  = flat.std(axis=(0, 2)) + 1e-8
    return mean, std


def normalise(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, :, None]) / std[None, :, None]


# ---------------------------------------------------------------------------
# 4. Dataset (with optional Gaussian jitter augmentation)
# ---------------------------------------------------------------------------
class TMDDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False, jitter_std: float = 0.05):
        self.X          = torch.from_numpy(X)
        self.y          = torch.from_numpy(y)
        self.augment    = augment
        self.jitter_std = jitter_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment and self.jitter_std > 0:
            x = x + torch.randn_like(x) * self.jitter_std
        return x, self.y[idx]


def make_loaders(X_tr, y_tr, X_val, y_val, X_test, y_test,
                 batch_size, jitter_std, num_workers):
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers > 0))
    tl = DataLoader(TMDDataset(X_tr,   y_tr,   augment=True,  jitter_std=jitter_std),
                    batch_size=batch_size, shuffle=True,  **kw)
    vl = DataLoader(TMDDataset(X_val,  y_val,  augment=False),
                    batch_size=batch_size, shuffle=False, **kw)
    el = DataLoader(TMDDataset(X_test, y_test, augment=False),
                    batch_size=batch_size, shuffle=False, **kw)
    return tl, vl, el


# ---------------------------------------------------------------------------
# 5. LR schedule: linear warm-up + cosine decay
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# 6. Train / eval step
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, scaler, device,
              grad_clip=0.0, desc=""):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    iterable = tqdm(loader, desc=desc, leave=False, ncols=90,
                    unit="batch", dynamic_ncols=True) if tqdm else loader

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in iterable:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(scaler is not None)):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            bs = len(y_batch)
            total_loss += loss.item() * bs
            correct    += (logits.detach().argmax(1) == y_batch).sum().item()
            total      += bs

            if tqdm and hasattr(iterable, "set_postfix"):
                iterable.set_postfix(loss=f"{total_loss/total:.4f}",
                                     acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def full_predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for X_batch, y_batch in loader:
        preds.append(model(X_batch.to(device)).argmax(1).cpu().numpy())
        gts.append(y_batch.numpy())
    return np.concatenate(preds), np.concatenate(gts)


# ---------------------------------------------------------------------------
# 7. Report helpers
# ---------------------------------------------------------------------------
def print_report(y_true, y_pred, class_names, log):
    log.info("\nClassification report:\n%s",
             classification_report(y_true, y_pred,
                                   target_names=class_names, digits=4))


def save_cm(y_true, y_pred, class_names, path):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=class_names).plot(
        ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log    = get_logger(OUT_DIR / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Output dir : %s", OUT_DIR)
    log.info("Device     : %s", device)

    # ------------------------------------------------------------------
    # 1. Parse
    # ------------------------------------------------------------------
    log.info("━━━  [1/5] Parsing dataset  ━━━")
    t0 = time.perf_counter()
    train_data, train_labels = parse_dat(args.data_dir / "train.dat", log)
    test_data,  test_labels  = parse_dat(args.data_dir / "test.dat",  log)
    log.info("  done in %.1f s", time.perf_counter() - t0)

    le           = LabelEncoder().fit(train_labels)
    train_labels = le.transform(train_labels).astype(np.int64)
    test_labels  = le.transform(test_labels).astype(np.int64)
    num_classes  = len(le.classes_)
    c_names = (args.class_names[:num_classes]
               if len(args.class_names) >= num_classes
               else [str(c) for c in le.classes_])
    log.info("  classes (%d): %s", num_classes, c_names)

    # ------------------------------------------------------------------
    # 2. Sliding windows
    # ------------------------------------------------------------------
    log.info("━━━  [2/5] Sliding windows  ━━━")
    X_all,  y_all  = sliding_windows(train_data, train_labels,
                                     args.window, args.stride_train)
    X_test, y_test = sliding_windows(test_data,  test_labels,
                                     args.window, args.stride_test)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=args.val_split,
        random_state=args.seed, stratify=y_all,
    )
    log.info("  train %d  |  val %d  |  test %d", len(y_tr), len(y_val), len(y_test))

    # ------------------------------------------------------------------
    # 3. Normalise
    # ------------------------------------------------------------------
    log.info("━━━  [3/5] Normalising  ━━━")
    mean, std = compute_stats(X_tr)
    X_tr, X_val, X_test = (normalise(a, mean, std) for a in (X_tr, X_val, X_test))
    log.info("  means: %s", np.round(mean, 4))
    log.info("  stds : %s", np.round(std, 4))

    # Inverse-frequency class weights
    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (num_classes * counts),
                     dtype=torch.float32, device=device)
    log.info("  class weights: %s", np.round(w.cpu().numpy(), 3))

    # ------------------------------------------------------------------
    # 4. Data loaders
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = make_loaders(
        X_tr, y_tr, X_val, y_val, X_test, y_test,
        args.batch_size, args.jitter_std, args.num_workers,
    )

    # ------------------------------------------------------------------
    # 5. Model / optimiser / scheduler
    # ------------------------------------------------------------------
    log.info("━━━  [4/5] Building model  ━━━")
    model     = TMD_1DCNN(num_classes=num_classes, window_size=args.window).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = build_scheduler(optimizer, args.warmup, args.epochs)
    scaler    = GradScaler() if device.type == "cuda" else None

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  trainable parameters: %d", params)

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    log.info("━━━  [5/5] Training  ━━━")
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_ckpt = OUT_DIR / "best.pt"
    last_ckpt = OUT_DIR / "last.pt"

    hdr = (f"{'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  "
           f"{'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}  Time")
    log.info(hdr)
    log.info("─" * len(hdr))

    for epoch in range(1, args.epochs + 1):
        t_ep = time.perf_counter()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, scaler, device,
                                    grad_clip=args.grad_clip,
                                    desc=f"Ep {epoch:3d}/{args.epochs} train")
        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    None, None, device,
                                    desc=f"Ep {epoch:3d}/{args.epochs} val  ")
        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t_ep

        log.info(
            f"{epoch:4d}  {tr_loss:8.4f}  {tr_acc:7.4f}  "
            f"{va_loss:8.4f}  {va_acc:7.4f}  {lr_now:9.2e}  {elapsed:.1f}s"
        )
        # last checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "mean": mean, "std": std, "le": le.classes_}, last_ckpt)

        # best checkpoint
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_ctr  = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "mean": mean, "std": std, "le": le.classes_}, best_ckpt)
            log.info("  ↳ best val-loss %.4f  →  saved %s", best_val_loss, best_ckpt.name)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                log.info("  Early stopping triggered at epoch %d", epoch)
                break

    # ------------------------------------------------------------------
    # 7. Final evaluation
    # ------------------------------------------------------------------
    log.info("━━━  Final evaluation on test set  ━━━")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False)["model"])
    model.to(device)
    te_loss, te_acc = run_epoch(model, test_loader, criterion,
                                None, None, device, desc="Test eval")
    log.info("  Test Loss : %.4f", te_loss)
    log.info("  Test Acc  : %.4f", te_acc)

    y_pred, y_true = full_predict(model, test_loader, device)
    print_report(y_true, y_pred, c_names, log)
    save_cm(y_true, y_pred, c_names, OUT_DIR / "confusion_matrix.png")
    if HAS_MATPLOTLIB:
        log.info("  Confusion matrix → %s", OUT_DIR / "confusion_matrix.png")

    # ------------------------------------------------------------------
    # 8. QONNX export
    # ------------------------------------------------------------------
    log.info("━━━  Exporting to QONNX  ━━━")
    qonnx_path = OUT_DIR / "tmd_4bit.onnx"
    export_model_to_qonnx(model, qonnx_path,
                          window_size=args.window, device=device)
    log.info("  QONNX model saved → %s", qonnx_path)

    log.info("All artefacts saved in:  %s", OUT_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TMD 4-bit Brevitas 1D-CNN — train & evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",    type=Path,  default=DATA_DIR)
    parser.add_argument("--class_names", nargs="+",  default=CLASS_NAMES)
    parser.add_argument("--window",       type=int,   default=WINDOW_SIZE)
    parser.add_argument("--stride_train", type=int,   default=STRIDE_TRAIN)
    parser.add_argument("--stride_test",  type=int,   default=STRIDE_TEST)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--wd",          type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup",      type=int,   default=WARMUP_EPOCHS)
    parser.add_argument("--val_split",   type=float, default=VAL_SPLIT)
    parser.add_argument("--patience",    type=int,   default=PATIENCE)
    parser.add_argument("--jitter_std",  type=float, default=JITTER_STD)
    parser.add_argument("--grad_clip",   type=float, default=GRAD_CLIP)
    parser.add_argument("--num_workers", type=int,   default=NUM_WORKERS)
    parser.add_argument("--seed",        type=int,   default=SEED)
    main(parser.parse_args())

