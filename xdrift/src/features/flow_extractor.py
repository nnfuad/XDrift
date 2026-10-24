"""Flow extraction and feature engineering for XDrift protocol.

Implements:
- Robust packet parsing across Ethernet, Linux SLL, VLAN, and Raw IP (DLT 101)
- Bidirectional 5-tuple canonicalization (Section 4.2)
- Chronological packet sorting per 5-tuple
- 120-second inactivity timeout segmentation (Section 4.2)
- Directional IP packet length normalization S_t / 1500 (Channel 1, Section 5)
- Log inter-arrival time ln(1 + min(Delta t, 5)) (Channel 2, Section 5)
- Zero-padding up to 64 packets with sequence length L and validity mask M (Section 5.1)
"""

import os
import re
import socket
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import dpkt


def get_ip_layer_and_len(buf: bytes) -> Tuple[Optional[Any], int]:
    """Robustly extract IP layer and IP packet length from raw packet buffer.
    
    Supports:
    - Standard Ethernet (DLT 1 / EN10MB)
    - 802.1Q VLAN tagged Ethernet frames
    - Linux Cooked SLL (DLT 113)
    - Raw IPv4 / IPv6 (DLT 101 / RAW)
    - BSD Loopback / NULL (DLT 0)
    
    Per Section 5: True IP packet length, not Ethernet frame length, must be used.
    """
    if len(buf) < 20:
        return None, 0

    ip = None

    # 1. Try Ethernet
    try:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            ip = eth.data
        elif hasattr(eth.data, 'data') and isinstance(eth.data.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            # VLAN 802.1Q tagged frame
            ip = eth.data.data
    except Exception:
        pass

    # 2. Try Linux Cooked SLL
    if ip is None:
        try:
            sll = dpkt.sll.SLL(buf)
            if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                ip = sll.data
        except Exception:
            pass

    # 3. Try BSD Loopback / NULL (4-byte header)
    if ip is None and len(buf) > 24:
        try:
            raw_null = dpkt.ip.IP(buf[4:])
            if raw_null.v in (4, 6):
                ip = raw_null
        except Exception:
            pass

    # 4. Try raw IP (DLT 101 / RAW)
    if ip is None:
        try:
            raw_ip = dpkt.ip.IP(buf)
            if raw_ip.v in (4, 6):
                ip = raw_ip
        except Exception:
            pass

    if ip is None:
        return None, 0

    # Extract true IP packet length (bytes) from IP header
    ip_len = 0
    if isinstance(ip, dpkt.ip.IP):
        ip_len = ip.len
        # If TCP Segmentation Offload (TSO) or zero len, fallback to buffer length of IP
        if ip_len == 0 or ip_len > len(buf):
            ip_len = len(ip)
    elif isinstance(ip, dpkt.ip6.IP6):
        ip_len = ip.plen + 40
        if ip_len == 0 or ip_len > len(buf):
            ip_len = len(ip)
    else:
        ip_len = len(ip)

    return ip, int(ip_len)


def canonicalize_flow_endpoints(
    src_ip: bytes, src_port: int, dst_ip: bytes, dst_port: int, proto: int
) -> Tuple[Tuple[Tuple[bytes, int], Tuple[bytes, int], int], int]:
    """Canonicalize bidirectional endpoints according to Section 4.2.
    
    Endpoint tuples are (IP, port). Complete endpoint tuples are sorted
    together to avoid incorrectly pairing unrelated endpoints.
    
    Returns:
        canonical_key: ((epA_ip, epA_port), (epB_ip, epB_port), proto)
        direction: +1 if packet goes epA -> epB, -1 if epB -> epA.
    """
    ep_src = (src_ip, src_port)
    ep_dst = (dst_ip, dst_port)

    if ep_src <= ep_dst:
        canonical_key = (ep_src, ep_dst, proto)
        direction = 1
    else:
        canonical_key = (ep_dst, ep_src, proto)
        direction = -1

    return canonical_key, direction


def extract_flows_from_pcap(
    file_path: str,
    max_packets: int = 64,
    min_packets: int = 2,
    inactivity_timeout: float = 120.0,
    mtu_norm: float = 1500.0,
    max_iat: float = 5.0
) -> Dict[str, Any]:
    """Parse a PCAP/PCAPNG file and extract flow feature tensors with audit metrics.
    
    Returns a dict containing:
    - 'records': list of flow dicts (tensor, mask, seq_len, flow_id, capture_id)
    - 'audit': dict of packet and parsing metrics (total_packets, read_packets, rejected_packets)
    """
    raw_packets_by_flow: Dict[Any, List[Tuple[float, int, int]]] = {}

    total_packets = 0
    rejected_non_ip = 0
    rejected_non_transport = 0
    accepted_packets = 0

    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            f.seek(0)
            if magic == b'\n\r\r\n':
                pcap = dpkt.pcapng.Reader(f)
            else:
                pcap = dpkt.pcap.Reader(f)

            for ts, buf in pcap:
                total_packets += 1
                ip, ip_len = get_ip_layer_and_len(buf)
                if not ip or ip_len <= 0:
                    rejected_non_ip += 1
                    continue

                if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    rejected_non_transport += 1
                    continue

                trans = ip.data
                src_ip = ip.src
                dst_ip = ip.dst
                src_port = trans.sport
                dst_port = trans.dport
                proto = ip.p

                canonical_key, direction = canonicalize_flow_endpoints(
                    src_ip, src_port, dst_ip, dst_port, proto
                )

                if canonical_key not in raw_packets_by_flow:
                    raw_packets_by_flow[canonical_key] = []

                raw_packets_by_flow[canonical_key].append((float(ts), direction, ip_len))
                accepted_packets += 1

    except Exception as e:
        return {
            'records': [],
            'audit': {
                'total_packets': total_packets,
                'accepted_packets': accepted_packets,
                'rejected_packets': total_packets - accepted_packets,
                'rejected_non_ip': rejected_non_ip,
                'rejected_non_transport': rejected_non_transport,
                'error': str(e)
            }
        }

    # Now process flows: sort by timestamp, then apply 120s inactivity timeout
    flow_records = []
    capture_id = os.path.splitext(os.path.basename(file_path))[0]

    for canonical_key, pkts in raw_packets_by_flow.items():
        ep_a, ep_b, proto = canonical_key
        try:
            src_str = f"{socket.inet_ntoa(ep_a[0])}:{ep_a[1]}"
            dst_str = f"{socket.inet_ntoa(ep_b[0])}:{ep_b[1]}"
        except Exception:
            src_str = f"{ep_a[0].hex()}:{ep_a[1]}"
            dst_str = f"{ep_b[0].hex()}:{ep_b[1]}"

        # Sort packets chronologically
        pkts.sort(key=lambda p: p[0])

        # Segment by 120-second inactivity timeout
        segments = [[]]
        prev_pkt_ts = None
        for pkt in pkts:
            pkt_ts = pkt[0]
            if prev_pkt_ts is not None and (pkt_ts - prev_pkt_ts) > inactivity_timeout:
                segments.append([])
            segments[-1].append(pkt)
            prev_pkt_ts = pkt_ts

        # Convert each segment to flow tensor
        for seg_idx, seg_pkts in enumerate(segments):
            if len(seg_pkts) < min_packets:
                continue

            actual_len = min(len(seg_pkts), max_packets)
            pkts_truncated = seg_pkts[:actual_len]

            tensor = np.zeros((2, max_packets), dtype=np.float32)
            mask = np.zeros(max_packets, dtype=np.float32)

            prev_ts = pkts_truncated[0][0]
            for i, (ts, direction, ip_len) in enumerate(pkts_truncated):
                # Channel 1: Directional packet length normalized to [-1, 1]
                norm_len = direction * min(float(ip_len) / mtu_norm, 1.0)
                tensor[0, i] = norm_len

                # Channel 2: Inter-arrival time (natural log)
                if i == 0:
                    iat = 0.0
                else:
                    delta_t = max(0.0, ts - prev_ts)
                    iat = np.log(1.0 + min(delta_t, max_iat))
                tensor[1, i] = iat

                prev_ts = ts
                mask[i] = 1.0

            flow_id = f"{capture_id}_{src_str}_{dst_str}_p{proto}_s{seg_idx}"
            flow_records.append({
                'tensor': tensor,
                'mask': mask,
                'seq_len': actual_len,
                'flow_id': flow_id,
                'capture_id': capture_id
            })

    audit = {
        'total_packets': total_packets,
        'accepted_packets': accepted_packets,
        'rejected_packets': total_packets - accepted_packets,
        'rejected_non_ip': rejected_non_ip,
        'rejected_non_transport': rejected_non_transport,
        'flows_generated': len(flow_records),
        'error': None
    }

    return {'records': flow_records, 'audit': audit}
