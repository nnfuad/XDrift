"""Predictive Evaluation Pipeline for FP32, PTQ, and QAT Models (Section 10).

Evaluates all models on identical held-out test examples.
Saves:
- Per-example predictions, logits, probabilities, and paired deltas
- Aggregate predictive metrics (accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, Brier score)
- Paired differences from FP32 (logit differences, probability differences, prediction flips, error introduction/correction)
"""

import os
import json
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.ao.quantization as quantization
from sklearn.metrics import confusion_matrix, brier_score_loss

from src.models.cnn1d import Compact1DCNN, fuse_model_ptq
from src.training.train import load_dataset, compute_metrics


def load_int8_model(model_path: str) -> torch.nn.Module:
    """Instantiate and load a converted INT8 model from saved state dict."""
    torch.backends.quantized.engine = 'qnnpack'
    payload = torch.load(model_path, map_location='cpu')
    
    # Recreate quantized architecture
    base_model = Compact1DCNN().eval()
    fused_model = fuse_model_ptq(base_model)
    fused_model.qconfig = quantization.get_default_qconfig('qnnpack')
    prep_model = quantization.prepare(fused_model)
    
    # Dummy forward to set up observer types before convert
    dummy_x = torch.zeros(1, 2, 64)
    prep_model(dummy_x)
    int8_model = quantization.convert(prep_model)
    
    int8_model.load_state_dict(payload['state_dict'])
    int8_model.eval()
    return int8_model


def run_model_inference(
    model: torch.nn.Module,
    X: torch.Tensor,
    batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model forward pass and return raw logits and probabilities."""
    model.eval()
    logits_list = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = X[i:i + batch_size]
            out = model(bx)
            logits_list.append(out.cpu().numpy())
            
    logits = np.vstack(logits_list).flatten()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs


def evaluate_all_models(
    test_npz_path: str,
    fp32_checkpoint_path: str,
    ptq_model_path: str,
    qat_model_path: Optional[str],
    results_dir: str
) -> Dict[str, Any]:
    """Evaluate FP32, PTQ, and QAT on the untouched test split."""
    os.makedirs(results_dir, exist_ok=True)
    predictions_dir = os.path.join(results_dir, "predictions")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Load test data
    X_test, masks_test, y_test_t, flow_ids, capture_ids = load_dataset(test_npz_path)
    y_test = y_test_t.numpy().flatten().astype(int)
    print(f"Loaded test set: {len(y_test)} examples (VPN={np.sum(y_test == 1)}, Non-VPN={np.sum(y_test == 0)})")

    # 2. Load models
    # FP32
    fp32_ckpt = torch.load(fp32_checkpoint_path, map_location='cpu')
    fp32_model = Compact1DCNN()
    fp32_model.load_state_dict(fp32_ckpt['state_dict'])
    fp32_model.eval()
    val_threshold = float(fp32_ckpt.get('validation_threshold', 0.5))

    # INT8-PTQ
    ptq_model = load_int8_model(ptq_model_path)

    # INT8-QAT (if provided)
    qat_model = None
    if qat_model_path and os.path.exists(qat_model_path):
        qat_model = load_int8_model(qat_model_path)

    # 3. Predict across all models
    fp32_logits, fp32_probs = run_model_inference(fp32_model, X_test)
    ptq_logits, ptq_probs = run_model_inference(ptq_model, X_test)

    fp32_preds = (fp32_probs >= val_threshold).astype(int)
    ptq_preds = (ptq_probs >= val_threshold).astype(int)

    # 4. Metrics
    fp32_metrics = compute_metrics(y_test, fp32_probs, threshold=val_threshold)
    fp32_metrics['brier_score'] = float(brier_score_loss(y_test, fp32_probs))
    fp32_metrics['confusion_matrix'] = confusion_matrix(y_test, fp32_preds).tolist()

    ptq_metrics = compute_metrics(y_test, ptq_probs, threshold=val_threshold)
    ptq_metrics['brier_score'] = float(brier_score_loss(y_test, ptq_probs))
    ptq_metrics['confusion_matrix'] = confusion_matrix(y_test, ptq_preds).tolist()

    # Paired comparisons (FP32 vs PTQ)
    logit_diff_ptq = ptq_logits - fp32_logits
    prob_diff_ptq = ptq_probs - fp32_probs
    flips_ptq = fp32_preds != ptq_preds
    errors_introduced_ptq = (fp32_preds == y_test) & (ptq_preds != y_test)
    errors_corrected_ptq = (fp32_preds != y_test) & (ptq_preds == y_test)

    paired_ptq = {
        'mean_logit_diff': float(np.mean(logit_diff_ptq)),
        'std_logit_diff': float(np.std(logit_diff_ptq)),
        'mean_abs_prob_diff': float(np.mean(np.abs(prob_diff_ptq))),
        'prediction_flip_rate': float(np.mean(flips_ptq)),
        'total_flips': int(np.sum(flips_ptq)),
        'errors_introduced': int(np.sum(errors_introduced_ptq)),
        'errors_corrected': int(np.sum(errors_corrected_ptq)),
        'delta_accuracy': float(ptq_metrics['accuracy'] - fp32_metrics['accuracy']),
        'delta_f1': float(ptq_metrics['f1'] - fp32_metrics['f1']),
        'delta_pr_auc': float(ptq_metrics['pr_auc'] - fp32_metrics['pr_auc'])
    }

    eval_summary = {
        'validation_threshold': val_threshold,
        'test_examples': len(y_test),
        'fp32_metrics': fp32_metrics,
        'ptq_metrics': ptq_metrics,
        'paired_ptq_diff': paired_ptq
    }

    # Evaluate QAT if present
    if qat_model:
        qat_logits, qat_probs = run_model_inference(qat_model, X_test)
        qat_preds = (qat_probs >= val_threshold).astype(int)
        qat_metrics = compute_metrics(y_test, qat_probs, threshold=val_threshold)
        qat_metrics['brier_score'] = float(brier_score_loss(y_test, qat_probs))
        qat_metrics['confusion_matrix'] = confusion_matrix(y_test, qat_preds).tolist()

        logit_diff_qat = qat_logits - fp32_logits
        prob_diff_qat = qat_probs - fp32_probs
        flips_qat = fp32_preds != qat_preds
        errors_introduced_qat = (fp32_preds == y_test) & (qat_preds != y_test)
        errors_corrected_qat = (fp32_preds != y_test) & (qat_preds == y_test)

        paired_qat = {
            'mean_logit_diff': float(np.mean(logit_diff_qat)),
            'std_logit_diff': float(np.std(logit_diff_qat)),
            'mean_abs_prob_diff': float(np.mean(np.abs(prob_diff_qat))),
            'prediction_flip_rate': float(np.mean(flips_qat)),
            'total_flips': int(np.sum(flips_qat)),
            'errors_introduced': int(np.sum(errors_introduced_qat)),
            'errors_corrected': int(np.sum(errors_corrected_qat)),
            'delta_accuracy': float(qat_metrics['accuracy'] - fp32_metrics['accuracy']),
            'delta_f1': float(qat_metrics['f1'] - fp32_metrics['f1']),
            'delta_pr_auc': float(qat_metrics['pr_auc'] - fp32_metrics['pr_auc'])
        }
        eval_summary['qat_metrics'] = qat_metrics
        eval_summary['paired_qat_diff'] = paired_qat
    else:
        qat_logits = None
        qat_probs = None
        qat_preds = None

    # Save per-example predictions NPZ
    pred_save_dict = {
        'y_true': y_test,
        'flow_ids': flow_ids,
        'capture_ids': capture_ids,
        'fp32_logits': fp32_logits,
        'fp32_probs': fp32_probs,
        'fp32_preds': fp32_preds,
        'ptq_logits': ptq_logits,
        'ptq_probs': ptq_probs,
        'ptq_preds': ptq_preds,
    }
    if qat_model:
        pred_save_dict.update({
            'qat_logits': qat_logits,
            'qat_probs': qat_probs,
            'qat_preds': qat_preds
        })

    pred_file = os.path.join(predictions_dir, "test_predictions.npz")
    np.savez_compressed(pred_file, **pred_save_dict)

    # Save metrics JSON
    summary_file = os.path.join(metrics_dir, "predictive_evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nSaved test predictions to {pred_file}")
    print(f"Saved evaluation metrics to {summary_file}")
    print(f"  FP32 Test: Acc={fp32_metrics['accuracy']:.4f}, BalAcc={fp32_metrics['balanced_accuracy']:.4f}, F1={fp32_metrics['f1']:.4f}, PR-AUC={fp32_metrics['pr_auc']:.4f}")
    print(f"  PTQ  Test: Acc={ptq_metrics['accuracy']:.4f}, BalAcc={ptq_metrics['balanced_accuracy']:.4f}, F1={ptq_metrics['f1']:.4f}, PR-AUC={ptq_metrics['pr_auc']:.4f}")
    print(f"  PTQ Flips: {paired_ptq['total_flips']}/{len(y_test)} ({paired_ptq['prediction_flip_rate']*100:.2f}%)")

    return eval_summary
