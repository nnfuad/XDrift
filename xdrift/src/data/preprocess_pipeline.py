"""End-to-end preprocessing pipeline for XDrift dataset.

Processes raw PCAP/PCAPNG files based on immutable split manifests,
extracts 2-channel flow tensors with validity masks and sequence lengths,
tracks packet-level audit statistics, and saves train_data.npz, val_data.npz, test_data.npz.
"""

import os
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm

from src.features.flow_extractor import extract_flows_from_pcap


def process_single_file(file_info: Dict[str, Any], raw_base_dir: str) -> Dict[str, Any]:
    """Worker task to process a single capture file and return audit data."""
    rel_path = file_info['relative_path']
    full_path = os.path.join(raw_base_dir, rel_path)
    label = file_info['label']
    capture_id = file_info['capture_session_id']
    file_format = file_info.get('format', 'unknown')

    if not os.path.exists(full_path):
        return {
            'status': 'error',
            'file': rel_path,
            'format': file_format,
            'label': label,
            'msg': 'File not found',
            'audit': {}
        }

    try:
        res = extract_flows_from_pcap(full_path)
        return {
            'status': 'success',
            'file': rel_path,
            'format': file_format,
            'label': label,
            'capture_id': capture_id,
            'records': res['records'],
            'audit': res['audit']
        }
    except Exception as e:
        return {
            'status': 'error',
            'file': rel_path,
            'format': file_format,
            'label': label,
            'msg': str(e),
            'audit': {}
        }


def process_split(
    manifest_split_path: str,
    raw_base_dir: str,
    output_npz_path: str,
    max_workers: int = 4,
    limit_files: int = 0
) -> Dict[str, Any]:
    """Process all files in a split manifest and save to compressed NPZ."""
    with open(manifest_split_path, 'r') as f:
        split_data = json.load(f)

    file_records = split_data['files']
    if limit_files > 0:
        file_records = file_records[:limit_files]

    print(f"\nProcessing {len(file_records)} files from {os.path.basename(manifest_split_path)}...")

    t0 = time.time()
    all_tensors = []
    all_masks = []
    all_seq_lens = []
    all_labels = []
    all_flow_ids = []
    all_capture_ids = []

    successful_files = 0
    failed_files = 0
    total_packets_read = 0
    total_packets_rejected = 0
    per_file_audits = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, rec, raw_base_dir): idx
            for idx, rec in enumerate(file_records)
        }

        ordered_results = [None] * len(file_records)
        for fut in tqdm(as_completed(futures), total=len(futures), desc=os.path.basename(output_npz_path)):
            idx = futures[fut]
            ordered_results[idx] = fut.result()

    for res in ordered_results:
        if res['status'] == 'success':
            successful_files += 1
            audit = res['audit']
            total_packets_read += audit.get('total_packets', 0)
            total_packets_rejected += audit.get('rejected_packets', 0)
            per_file_audits.append({
                'file': res['file'],
                'format': res['format'],
                'label': res['label'],
                'flows': len(res['records']),
                'total_packets': audit.get('total_packets', 0),
                'accepted_packets': audit.get('accepted_packets', 0),
                'rejected_packets': audit.get('rejected_packets', 0)
            })

            for rec in res['records']:
                all_tensors.append(rec['tensor'])
                all_masks.append(rec['mask'])
                all_seq_lens.append(rec['seq_len'])
                all_labels.append(res['label'])
                all_flow_ids.append(rec['flow_id'])
                all_capture_ids.append(res['capture_id'])
        else:
            failed_files += 1
            per_file_audits.append({
                'file': res['file'],
                'format': res['format'],
                'label': res['label'],
                'error': res.get('msg', 'Unknown error')
            })

    t1 = time.time()
    n_flows = len(all_tensors)
    print(f"Extraction complete in {t1-t0:.2f}s. Extracted {n_flows} flows ({successful_files} ok, {failed_files} failed).")

    if n_flows == 0:
        raise RuntimeError(f"No flows extracted for {output_npz_path}!")

    X_arr = np.stack(all_tensors, axis=0).astype(np.float32)
    masks_arr = np.stack(all_masks, axis=0).astype(np.float32)
    seq_lens_arr = np.array(all_seq_lens, dtype=np.int32)
    y_arr = np.array(all_labels, dtype=np.int64)
    flow_ids_arr = np.array(all_flow_ids, dtype=object)
    capture_ids_arr = np.array(all_capture_ids, dtype=object)

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        X=X_arr,
        masks=masks_arr,
        seq_lens=seq_lens_arr,
        y=y_arr,
        flow_ids=flow_ids_arr,
        capture_ids=capture_ids_arr
    )

    stats = {
        'split': os.path.basename(output_npz_path),
        'files_processed': successful_files,
        'files_failed': failed_files,
        'total_packets_read': total_packets_read,
        'total_packets_rejected': total_packets_rejected,
        'total_flows': int(n_flows),
        'vpn_flows': int(np.sum(y_arr == 1)),
        'nonvpn_flows': int(np.sum(y_arr == 0)),
        'vpn_ratio': float(np.mean(y_arr == 1)),
        'seq_len_mean': float(np.mean(seq_lens_arr)),
        'seq_len_median': float(np.median(seq_lens_arr)),
        'seq_len_min': int(np.min(seq_lens_arr)),
        'seq_len_max': int(np.max(seq_lens_arr)),
        'tensor_shape': list(X_arr.shape),
        'duration_seconds': round(t1 - t0, 2),
        'per_file_audits': per_file_audits
    }
    print(f"Saved {output_npz_path}: shape {X_arr.shape}, VPN={stats['vpn_flows']}, Non-VPN={stats['nonvpn_flows']} ({stats['vpn_ratio']*100:.2f}%)")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess XDrift Splits")
    parser.add_argument('--raw_dir', type=str, default="/Volumes/LEXAR/Explainability Drift/Dataset/ISCX VPN-NonVPN 2016 Dataset")
    parser.add_argument('--manifest_dir', type=str, default="/Volumes/LEXAR/Explainability Drift/xdrift/data/manifests")
    parser.add_argument('--output_dir', type=str, default="/Volumes/LEXAR/Explainability Drift/xdrift/data/processed")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit_files', type=int, default=0, help="Limit number of files per split (for testing/pilots)")
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    all_stats = {}

    for s in splits:
        manifest_path = os.path.join(args.manifest_dir, f"{s}_manifest.json")
        out_npz = os.path.join(args.output_dir, f"{s}_data.npz")
        all_stats[s] = process_split(
            manifest_path, args.raw_dir, out_npz,
            max_workers=args.workers, limit_files=args.limit_files
        )

    stats_file = os.path.join(args.output_dir, "preprocessing_summary.json")
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nAll splits preprocessed successfully. Summary saved to {stats_file}")


if __name__ == '__main__':
    main()
