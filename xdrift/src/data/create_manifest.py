"""Dataset manifest generator for ISCX VPN-NonVPN 2016 dataset."""

import os
import re
import json
import hashlib
from typing import Dict, Any, List

def extract_app_label(filename: str) -> str:
    """Extract application category from capture filename."""
    base = os.path.basename(filename)
    base = re.sub(r'(?i)\.pcapng?$', '', base)
    base = re.sub(r'(?i)^vpn_', '', base)
    match = re.search(r'([a-zA-Z]+)', base)
    if match:
        return match.group(1).lower()
    return 'unknown'

def compute_quick_hash(file_path: str, max_bytes: int = 1024 * 1024) -> str:
    """Compute sha256 hash of header and first 1MB of file for practicality."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        chunk = f.read(max_bytes)
        hasher.update(chunk)
    return hasher.hexdigest()

def generate_manifest(raw_dir: str, output_path: str) -> Dict[str, Any]:
    raw_dir = os.path.abspath(raw_dir)
    manifest_records = []
    
    vpn_dirs = {'VPN-PCAPS-01', 'VPN-PCAPS-02', 'VPN-PCAP-01', 'VPN-PCAP-02'}
    
    total_bytes = 0
    vpn_count = 0
    nonvpn_count = 0
    
    for root, _, files in os.walk(raw_dir):
        rel_dir = os.path.relpath(root, raw_dir)
        parent_name = os.path.basename(root).upper()
        
        for f in sorted(files):
            if f.startswith('.') or not f.lower().endswith(('.pcap', '.pcapng')):
                continue
            
            full_path = os.path.join(root, f)
            size = os.path.getsize(full_path)
            total_bytes += size
            
            is_vpn = 1 if ('VPN' in parent_name and 'NONVPN' not in parent_name) else 0
            if is_vpn:
                vpn_count += 1
            else:
                nonvpn_count += 1
                
            fmt = 'pcapng' if f.lower().endswith('.pcapng') else 'pcap'
            app = extract_app_label(f)
            quick_hash = compute_quick_hash(full_path)
            
            manifest_records.append({
                'filename': f,
                'relative_path': os.path.join(rel_dir, f),
                'directory': os.path.basename(root),
                'format': fmt,
                'label': is_vpn,
                'label_name': 'VPN' if is_vpn == 1 else 'Non-VPN',
                'application': app,
                'file_size_bytes': size,
                'header_sha256': quick_hash,
                'capture_session_id': os.path.splitext(f)[0]
            })
            
    manifest = {
        'total_files': len(manifest_records),
        'total_bytes': total_bytes,
        'total_size_gb': round(total_bytes / (1024**3), 2),
        'vpn_files': vpn_count,
        'nonvpn_files': nonvpn_count,
        'records': manifest_records
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Generated manifest with {len(manifest_records)} files ({vpn_count} VPN, {nonvpn_count} Non-VPN) saved to {output_path}")
    return manifest

if __name__ == '__main__':
    raw_dir = "/Volumes/LEXAR/Explainability Drift/Dataset/ISCX VPN-NonVPN 2016 Dataset"
    output_path = "/Volumes/LEXAR/Explainability Drift/xdrift/data/manifests/dataset_manifest.json"
    generate_manifest(raw_dir, output_path)
