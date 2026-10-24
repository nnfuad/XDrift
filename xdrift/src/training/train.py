"""Training pipeline for Compact1DCNN baseline models (Section 8).

Features:
- Fixed, documented training procedure with AdamW, lr=1e-3, weight_decay=1e-4
- Training-only class-weighted BCE loss (pos_weight = N_neg / N_pos)
- Early stopping based strictly on validation PR-AUC / loss (patience = 10)
- Best checkpoint selection from validation split only
- Prespecified multi-seed training support ([42, 123, 456, 789, 1011])
- Model and training history artifact persistence
"""

import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

from src.models.cnn1d import Compact1DCNN


def set_seed(seed: int) -> None:
    """Set deterministic random seeds across PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(npz_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load preprocessed tensors, masks, labels, flow IDs, and capture IDs."""
    data = np.load(npz_path, allow_pickle=True)
    X = torch.from_numpy(data['X']).float()
    masks = torch.from_numpy(data['masks']).float()
    y = torch.from_numpy(data['y']).float().unsqueeze(1)
    flow_ids = data['flow_ids']
    capture_ids = data['capture_ids']
    return X, masks, y, flow_ids, capture_ids


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Check if both classes are present for AUC calculations
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = float(np.mean(y_true == y_pred))

    # Balanced accuracy
    rec_0 = float(np.sum((y_true == 0) & (y_pred == 0)) / max(1, np.sum(y_true == 0)))
    rec_1 = float(rec)
    balanced_acc = 0.5 * (rec_0 + rec_1)

    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'minority_recall': rec_1
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the optimal threshold on validation data maximizing F1-score."""
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    return best_thresh


def train_single_seed(
    train_npz: str,
    val_npz: str,
    output_checkpoint_dir: str,
    seed: int = 42,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 10,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Train FP32 Compact1DCNN under a single seed."""
    set_seed(seed)
    dev = torch.device(device)

    # 1. Load data
    X_train, masks_train, y_train, _, _ = load_dataset(train_npz)
    X_val, masks_val, y_val, _, _ = load_dataset(val_npz)

    n_pos = float(torch.sum(y_train == 1))
    n_neg = float(torch.sum(y_train == 0))
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)]).to(dev)
    print(f"Seed {seed}: Train size={len(y_train)} (VPN={int(n_pos)}, Non-VPN={int(n_neg)}). Pos weight={pos_weight.item():.4f}")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate model, loss, optimizer
    model = Compact1DCNN().to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Training loop with early stopping
    best_val_loss = float('inf')
    best_val_pr_auc = -1.0
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history = []

    os.makedirs(output_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_checkpoint_dir, f"fp32_seed_{seed}.pt")

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(1, train_batches)

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_logits = []
        all_val_targets = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(dev), by.to(dev)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()
                val_batches += 1
                all_val_logits.append(logits.cpu().numpy())
                all_val_targets.append(by.cpu().numpy())

        val_loss /= max(1, val_batches)
        val_logits = np.vstack(all_val_logits).flatten()
        val_targets = np.vstack(all_val_targets).flatten()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        val_metrics = compute_metrics(val_targets, val_probs)
        val_pr_auc = val_metrics['pr_auc']

        epoch_record = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_pr_auc': float(val_pr_auc),
            'val_f1': float(val_metrics['f1']),
            'val_balanced_acc': float(val_metrics['balanced_accuracy'])
        }
        history.append(epoch_record)

        # Early stopping criterion on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val PR-AUC={val_pr_auc:.4f}, Val F1={val_metrics['f1']:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} (val_loss={best_val_loss:.4f}).")
            break

    train_time = time.time() - t0

    # 4. Save best checkpoint
    model.load_state_dict(best_state)
    model.to(dev)
    model.eval()

    # Determine validation-optimal threshold using the best model
    all_val_logits = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(dev)
            logits = model(bx)
            all_val_logits.append(logits.cpu().numpy())
    val_logits = np.vstack(all_val_logits).flatten()
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    best_threshold = find_best_threshold(val_targets, val_probs)

    checkpoint_payload = {
        'seed': seed,
        'model_architecture': 'Compact1DCNN',
        'parameter_count': model.count_parameters(),
        'state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_pr_auc': float(best_val_pr_auc),
        'validation_threshold': float(best_threshold),
        'pos_weight': float(pos_weight.item()),
        'training_time_seconds': round(train_time, 2),
        'history': history
    }

    torch.save(checkpoint_payload, checkpoint_path)
    print(f"Saved best checkpoint to {checkpoint_path} (threshold={best_threshold:.3f})")

    return checkpoint_payload


def main():
    parser = argparse.ArgumentParser(description="Train FP32 1D-CNN Models")
    parser.add_argument('--train_npz', type=str, default="data/processed/train_data.npz")
    parser.add_argument('--val_npz', type=str, default="data/processed/val_data.npz")
    parser.add_argument('--output_dir', type=str, default="artifacts/checkpoints")
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 1011])
    parser.add_argument('--pilot', action='store_true', help="Run single pilot seed (42)")
    args = parser.parse_args()

    seeds = [42] if args.pilot else args.seeds
    print(f"Starting training run on {args.device} for seeds: {seeds}")

    summary = {}
    for s in seeds:
        print(f"\n================ Training Seed {s} ================")
        res = train_single_seed(
            train_npz=args.train_npz,
            val_npz=args.val_npz,
            output_checkpoint_dir=args.output_dir,
            seed=s,
            device=args.device
        )
        summary[f"seed_{s}"] = {
            'best_epoch': res['best_epoch'],
            'best_val_loss': res['best_val_loss'],
            'best_val_pr_auc': res['best_val_pr_auc'],
            'validation_threshold': res['validation_threshold'],
            'training_time_seconds': res['training_time_seconds']
        }

    summary_file = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining complete. Summary saved to {summary_file}")


if __name__ == '__main__':
    main()
