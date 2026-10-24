"""Leakage-free dataset splitting module for XDrift protocol.

Ensures complete isolation between Train, Validation, and Test splits
by partitioning at the capture/session group level (Section 4.3).
"""

import os
import re
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any


# Documented alias map for verified typographical naming asymmetries in ISCX 2016 dataset.
# In NonVPN-PCAPs-02, Hangouts Chat Session 4 was recorded across two machines where
# Endpoint A was saved as 'hangouts_chat_4a.pcap' (plural) and Endpoint B was saved as
# 'hangout_chat_4b.pcap' (singular typo). Both endpoints share timestamps within 3.45s
# (recorded April 29, 2015, ~17:20 UTC) and communicating IP subnets (131.202.240.87 <-> 131.202.240.242),
# confirming they are the two communicating endpoints of the same physical recording session.
EXPLICIT_SESSION_ALIASES: Dict[str, str] = {
    'hangout_chat_4': 'hangouts_chat_4',
}


def get_session_group(filename: str) -> str:
    """Group capture files by collection session activity.
    
    Pairs like `facebook_audio1a.pcap` and `facebook_audio1b.pcapng`
    represent two endpoints of the same capture session and must share
    the same group ID to prevent leakage.
    
    Also normalizes verified dataset typographical naming asymmetries (e.g.,
    `hangout_chat_4b` -> `hangouts_chat_4`).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # 1. Remove trailing single letter or _a / _b / _A / _B
    group = re.sub(r'(?i)[_]?[a-b]$', '', base).lower()
    # 2. Apply verified explicit alias normalization
    group = EXPLICIT_SESSION_ALIASES.get(group, group)
    return group


def create_group_stratified_split(
    manifest_records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Partition capture files into Train, Val, and Test by session groups.
    
    Returns lists of filenames for (train, val, test).
    Guarantees:
    - Zero session group overlap between train, val, and test.
    - Balanced stratification of VPN vs Non-VPN classes.
    """
    # 1. Group files by session and label
    groups_by_label = defaultdict(lambda: defaultdict(list))
    for rec in manifest_records:
        grp = get_session_group(rec['filename'])
        label = rec['label']
        groups_by_label[label][grp].append(rec['filename'])

    rng = random.Random(random_seed)
    train_files = []
    val_files = []
    test_files = []

    # Stratify by class label
    for label, grp_dict in sorted(groups_by_label.items()):
        group_keys = sorted(grp_dict.keys())
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_train = max(1, int(round(n_groups * train_ratio)))
        n_val = max(1, int(round(n_groups * val_ratio)))
        # Remaining go to test to ensure exact sum
        train_grps = set(group_keys[:n_train])
        val_grps = set(group_keys[n_train:n_train + n_val])
        test_grps = set(group_keys[n_train + n_val:])

        # If test_grps is empty due to small group count, reallocate
        if len(test_grps) == 0 and len(train_grps) > 1:
            popped = train_grps.pop()
            test_grps.add(popped)

        for grp in train_grps:
            train_files.extend(grp_dict[grp])
        for grp in val_grps:
            val_files.extend(grp_dict[grp])
        for grp in test_grps:
            test_files.extend(grp_dict[grp])

    # Verification: check for any group overlap
    train_groups = {get_session_group(f) for f in train_files}
    val_groups = {get_session_group(f) for f in val_files}
    test_groups = {get_session_group(f) for f in test_files}

    assert len(train_groups & val_groups) == 0, f"Leakage: train & val overlap {train_groups & val_groups}"
    assert len(train_groups & test_groups) == 0, f"Leakage: train & test overlap {train_groups & test_groups}"
    assert len(val_groups & test_groups) == 0, f"Leakage: val & test overlap {val_groups & test_groups}"

    return sorted(train_files), sorted(val_files), sorted(test_files)


def save_split_manifests(
    manifest_records: List[Dict[str, Any]],
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Create and save immutable split manifests to disk."""
    train_files, val_files, test_files = create_group_stratified_split(
        manifest_records, train_ratio, val_ratio, test_ratio, random_seed
    )

    rec_map = {r['filename']: r for r in manifest_records}
    
    train_records = [rec_map[f] for f in train_files]
    val_records = [rec_map[f] for f in val_files]
    test_records = [rec_map[f] for f in test_files]

    splits = {
        'train': {
            'count': len(train_records),
            'vpn_count': sum(1 for r in train_records if r['label'] == 1),
            'nonvpn_count': sum(1 for r in train_records if r['label'] == 0),
            'files': train_records
        },
        'val': {
            'count': len(val_records),
            'vpn_count': sum(1 for r in val_records if r['label'] == 1),
            'nonvpn_count': sum(1 for r in val_records if r['label'] == 0),
            'files': val_records
        },
        'test': {
            'count': len(test_records),
            'vpn_count': sum(1 for r in test_records if r['label'] == 1),
            'nonvpn_count': sum(1 for r in test_records if r['label'] == 0),
            'files': test_records
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train_manifest.json'), 'w') as f:
        json.dump(splits['train'], f, indent=2)
    with open(os.path.join(output_dir, 'val_manifest.json'), 'w') as f:
        json.dump(splits['val'], f, indent=2)
    with open(os.path.join(output_dir, 'test_manifest.json'), 'w') as f:
        json.dump(splits['test'], f, indent=2)

    print(f"Saved split manifests to {output_dir}:")
    print(f"  Train: {len(train_records)} files ({splits['train']['vpn_count']} VPN, {splits['train']['nonvpn_count']} Non-VPN)")
    print(f"  Val:   {len(val_records)} files ({splits['val']['vpn_count']} VPN, {splits['val']['nonvpn_count']} Non-VPN)")
    print(f"  Test:  {len(test_records)} files ({splits['test']['vpn_count']} VPN, {splits['test']['nonvpn_count']} Non-VPN)")

    return splits


if __name__ == '__main__':
    manifest_path = "/Volumes/LEXAR/Explainability Drift/xdrift/data/manifests/dataset_manifest.json"
    output_dir = "/Volumes/LEXAR/Explainability Drift/xdrift/data/manifests"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    save_split_manifests(manifest['records'], output_dir)
