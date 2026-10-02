import os
import re
import json
import time
import argparse
import logging
from collections import defaultdict, Counter
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import dpkt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

MAX_PACKETS = 64

def get_ip_layer(buf):
    """Robustly extract IP layer from a packet buffer."""
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return eth.data
    except Exception:
        pass
    try:
        sll = dpkt.sll.SLL(buf)
        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return sll.data
    except Exception:
        pass
    try:
        ip = dpkt.ip.IP(buf)
        if ip.v in (4, 6):
            return ip
    except Exception:
        pass
    return None

def extract_app_label(filename):
    """Extract application name from filename using regex."""
    base = os.path.basename(filename)
    base = re.sub(r'(?i)\.pcapng?$', '', base)
    base = re.sub(r'(?i)^vpn_', '', base)
    match = re.search(r'([a-zA-Z]+)', base)
    if match:
        return match.group(1).lower()
    return 'unknown'

def process_pcap(file_path):
    """Process a single PCAP/PCAPNG file and extract flow features."""
    flows = defaultdict(list)
    parent_dir = os.path.basename(os.path.dirname(file_path)).upper()
    y_vpn = 1 if 'VPN-PCAPS' in parent_dir else 0
    if y_vpn == 1 and 'NONVPN' in parent_dir:
        y_vpn = 0
    y_app = extract_app_label(file_path)
    
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            f.seek(0)
            if magic == b'\n\r\r\n':
                pcap = dpkt.pcapng.Reader(f)
            else:
                pcap = dpkt.pcap.Reader(f)
                
            for ts, buf in pcap:
                ip = get_ip_layer(buf)
                if not ip:
                    continue
                    
                if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    continue
                    
                trans = ip.data
                src_ip = ip.src
                dst_ip = ip.dst
                src_port = trans.sport
                dst_port = trans.dport
                proto = ip.p
                
                if (src_ip, src_port) < (dst_ip, dst_port):
                    flow_id = (src_ip, dst_ip, src_port, dst_port, proto)
                    direction = 1
                else:
                    flow_id = (dst_ip, src_ip, dst_port, src_port, proto)
                    direction = -1
                    
                length = len(buf)
                flows[flow_id].append((ts, direction, length))
    except Exception as e:
        return None, None, None, str(e)
        
    X_list = []
    
    for flow_id, pkts in flows.items():
        if len(pkts) < 4:
            continue
            
        pkts.sort(key=lambda x: x[0])
        pkts = pkts[:MAX_PACKETS]
        
        tensor = np.zeros((2, MAX_PACKETS), dtype=np.float32)
        prev_ts = pkts[0][0]
        
        for i, (ts, direction, length) in enumerate(pkts):
            norm_len = (length / 1500.0) * direction
            tensor[0, i] = max(min(norm_len, 1.0), -1.0)
            
            iat = ts - prev_ts
            iat = min(max(iat, 0.0), 5.0)
            tensor[1, i] = np.log10(1 + iat)
            
            prev_ts = ts
            
        X_list.append(tensor)
        
    if len(X_list) == 0:
        return None, None, None, "No valid flows found"
        
    X_arr = np.stack(X_list)
    return X_arr, y_vpn, y_app, None

def main():
    parser = argparse.ArgumentParser(description="PCAP Preprocessing Pipeline")
    parser.add_argument('--input_dir', type=str, default="Dataset/ISCX VPN-NonVPN 2016 Dataset", help="Input directory")
    parser.add_argument('--output_dir', type=str, default="Dataset/processed", help="Output directory")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of worker processes")
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pcap_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.pcap', '.pcapng')):
                pcap_files.append(os.path.join(root, f))
                
    print(f"Found {len(pcap_files)} PCAP files.")
    
    start_time = time.time()
    
    X_all = []
    y_vpn_all = []
    y_app_all = []
    processed = 0
    skipped = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pcap, f): f for f in pcap_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PCAPs"):
            X_arr, y_vpn, y_app, err = future.result()
            if err:
                skipped += 1
            else:
                X_all.append(X_arr)
                y_vpn_all.extend([y_vpn] * len(X_arr))
                y_app_all.extend([y_app] * len(X_arr))
                processed += 1
                
    if len(X_all) == 0:
        print("No valid data extracted.")
        return
        
    X = np.concatenate(X_all, axis=0)
    y_vpn = np.array(y_vpn_all, dtype=np.int8)
    y_app = np.array(y_app_all)
    
    print(f"Total flows extracted: {len(X)}")
    print(f"Features shape: {X.shape}")
    
    # Attempt stratified split on (y_vpn, y_app)
    strat_key = [f"{v}_{a}" for v, a in zip(y_vpn, y_app)]
    counts = Counter(strat_key)
    safe_strat_key = [k if counts[k] >= 10 else "rare" for k in strat_key]
    
    try:
        X_train, X_temp, y_vpn_train, y_vpn_temp, y_app_train, y_app_temp = train_test_split(
            X, y_vpn, y_app, test_size=0.3, random_state=42, stratify=safe_strat_key
        )
        
        # for val/test, fallback to y_vpn_temp if safe_strat_key fails
        strat_key_temp = [f"{v}_{a}" for v, a in zip(y_vpn_temp, y_app_temp)]
        counts_temp = Counter(strat_key_temp)
        safe_strat_key_temp = [k if counts_temp[k] >= 2 else "rare" for k in strat_key_temp]
        
        X_val, X_test, y_vpn_val, y_vpn_test, y_app_val, y_app_test = train_test_split(
            X_temp, y_vpn_temp, y_app_temp, test_size=0.5, random_state=42, stratify=safe_strat_key_temp
        )
    except Exception as e:
        print(f"Detailed stratified split failed ({e}), falling back to VPN-only stratification...")
        X_train, X_temp, y_vpn_train, y_vpn_temp, y_app_train, y_app_temp = train_test_split(
            X, y_vpn, y_app, test_size=0.3, random_state=42, stratify=y_vpn
        )
        X_val, X_test, y_vpn_val, y_vpn_test, y_app_val, y_app_test = train_test_split(
            X_temp, y_vpn_temp, y_app_temp, test_size=0.5, random_state=42, stratify=y_vpn_temp
        )

    # Save tensors
    np.savez_compressed(os.path.join(output_dir, 'train_data.npz'), X_train=X_train, y_vpn_train=y_vpn_train, y_app_train=y_app_train)
    np.savez_compressed(os.path.join(output_dir, 'val_data.npz'), X_val=X_val, y_vpn_val=y_vpn_val, y_app_val=y_app_val)
    np.savez_compressed(os.path.join(output_dir, 'test_data.npz'), X_test=X_test, y_vpn_test=y_vpn_test, y_app_test=y_app_test)

    # Manifest
    elapsed = time.time() - start_time
    vpn_count = int(np.sum(y_vpn))
    manifest = {
        "total_pcaps_processed": processed,
        "total_pcaps_skipped": skipped,
        "total_flows": len(X),
        "vpn_flows": vpn_count,
        "nonvpn_flows": len(X) - vpn_count,
        "vpn_ratio": round(vpn_count / len(X), 4) if len(X) > 0 else 0,
        "tensor_shape": list(X.shape),
        "execution_time_seconds": round(elapsed, 2)
    }
    
    with open(os.path.join(output_dir, 'dataset_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print("Pipeline completed successfully.")
    
if __name__ == '__main__':
    main()
