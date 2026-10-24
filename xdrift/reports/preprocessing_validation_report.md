# XDrift: Preprocessing & Group-Aware Split Validation Gate Report

**Date**: October 23, 2026  
**Pipeline**: Clean-room implementation of XDrift Protocol  
**Repository Path**: `xdrift/`  
**Gate Status**: **PASSED** (0 Blocking Issues, 3 Documented Non-Blocking Limitations)

---

## 1. Audit of the Group-Aware Split

### 1.1 Identification of Recording Groups
In `src/data/leakage_free_split.py`, recording sessions are identified by parsing capture filenames. Capture files recorded from paired endpoints (e.g., `facebook_audio1a.pcap` and `facebook_audio1b.pcapng`, `email1a.pcap` and `email1b.pcap`, `vpn_ftps_A.pcap` and `vpn_ftps_B.pcap`) represent two perspectives of the exact same network conversation. 

The canonical grouping rule extracts the base activity string by stripping terminal participant suffixes (`_a`, `_b`, `_A`, `_B`):
$$\text{SessionGroup}(f) = \text{regex\_sub}(\text{"(?i)[\_]?[a-b]\$"}, \text{""}, \text{basename}(f)).\text{lower}()$$

### 1.2 Verification of Group Integrity and Disjunction
We audited all 140 capture files in `data/manifests/dataset_manifest.json`:
- **Total Capture Files**: 140
- **Total Unique Session Groups**: 107
- **Label Conflict Audit**: **0 groups contain conflicting labels**. Every session group is 100% homogeneous with respect to the binary label (VPN vs Non-VPN).
- **Group Class Breakdown**:
  - VPN Session Groups: 20 groups (31 files)
  - Non-VPN Session Groups: 87 groups (109 files)

### 1.3 Split Manifest Verification
Partitioning was performed via class-stratified group allocation (70% Train, 15% Validation, 15% Test) using random seed 42.

| Split | Total Files | VPN Files | Non-VPN Files | Total Groups | VPN Groups | Non-VPN Groups |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 99 | 24 | 75 | 75 | 14 | 61 |
| **Validation** | 20 | 4 | 16 | 16 | 3 | 13 |
| **Test** | 21 | 3 | 18 | 16 | 3 | 13 |
| **Total** | **140** | **31** | **109** | **107** | **20** | **87** |

**Cross-Split Overlap Check**:
$$\text{TrainGroups} \cap \text{ValGroups} = \emptyset$$
$$\text{TrainGroups} \cap \text{TestGroups} = \emptyset$$
$$\text{ValGroups} \cap \text{TestGroups} = \emptyset$$
Zero session leakage across splits is mathematically guaranteed and verified.

### 1.4 Generalization and Statistical Inference Implications of the Test Split
The test split contains **3 VPN captures** (`vpn_bittorrent.pcap`, `vpn_hangouts_audio2.pcap`, `vpn_netflix_A.pcap`) and **18 Non-VPN captures** across 16 independent session groups.

1. **Generalization Scope**:
   - The 3 test VPN captures span three distinct network traffic archetypes: peer-to-peer file transfer (BitTorrent), interactive real-time voice (Hangouts Audio), and bulk adaptive video streaming (Netflix).
   - This provides rigorous testing of whether a 1D-CNN generalizes to unseen application sessions rather than memorizing IP/port artifacts from seen captures. However, external generalization cannot be claimed beyond these benchmark sessions.
2. **Statistical Inference & Cluster Bootstrap**:
   - With $G_{\text{test, VPN}} = 3$ clusters, resampling groups with replacement will produce high cluster-to-cluster variance.
   - Per Section 14.1, this is an inherent structural limitation of the ISCX 2016 dataset (which contains only 20 VPN recording sessions in total). Reallocating groups to test would starve the training set (reducing training VPN groups to 10–12).
   - **Protocol Decision**: Retain the 70/15/15 split. The statistical inference pipeline must report both paired flow-level bootstrap confidence intervals and group-clustered bootstrap intervals with an explicit statement regarding small cluster count.

---

## 2. Preprocessing Implementation Audit

The preprocessing pipeline (`src/features/flow_extractor.py`, `src/data/preprocess_pipeline.py`) was audited against Section 4 & 5 requirements:

1. **Packet Parsing & DLT Support**:
   - Handles Ethernet (DLT 1), 802.1Q VLAN tagged frames, Linux Cooked SLL (DLT 113), and Raw IP captures (DLT 101, which represents 28 files in the dataset).
   - Rejects non-IP packets (ARP, STP) and non-TCP/UDP packets (ESP, ICMP).
2. **Bidirectional 5-Tuple Canonicalization**:
   - Canonical endpoint ordering: complete endpoint tuples $(IP_A, port_A) \le (IP_B, port_B)$ are compared lexicographically without separating IP from port.
   - Packet direction $d_t = +1$ for forward, $-1$ for reverse.
3. **Packet Ordering & 120s Inactivity Timeout**:
   - Packets belonging to the same 5-tuple are chronologically sorted by capture timestamp before flow segmentation.
   - An inactivity gap exceeding $120.0$ seconds initiates a new flow segment.
4. **Channel Representations**:
   - **Channel 1 (Directional Length)**: $x_{\text{length}, t} = d_t \times \min(S_t / 1500.0, 1.0)$, using actual IP packet length ($S_t$), not Ethernet buffer length.
   - **Channel 2 (Inter-arrival Time)**: For $t > 0$, $x_{\text{IAT}, t} = \ln(1 + \min(\Delta t_t, 5.0))$, where $\Delta t_t = \max(0, \text{ts}_t - \text{ts}_{t-1})$. For $t=0$, $x_{\text{IAT}, 0} = 0.0$.
5. **Padding & Validity Masks**:
   - Sequences are right-padded to 64 packets with exact zeros.
   - Validity mask $M_t = 1$ for $t < L$ and $0$ for $t \ge L$.
6. **Leakage Prevention**:
   - Preprocessing consumes immutable split manifests. No parameters or normalization statistics are learned from validation or test data.

---

## 3. Representative Preprocessing Pilot Results

A multi-split, multi-format, multi-class representative cohort of **9 captures** was processed through the complete extraction pipeline:

### 3.1 Per-Capture Processing Breakdown

| Split | Capture File | Format | Label | Size (MB) | Packets Read | Packets Accepted | Rejected | Flows Gen. | SeqLen (Med / Max) | Duration (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `aim_chat_3a.pcap` | pcap | Non-VPN | 0.73 | 1,243 | 1,225 | 18 | 42 | 2.0 / 64 | 0.062 |
| **Train** | `AIMchat1.pcapng` | pcapng | Non-VPN | 0.06 | 404 | 384 | 20 | 29 | 2.0 / 64 | 0.010 |
| **Train** | `vpn_aim_chat1a.pcap` | pcap (RAW) | VPN | 0.17 | 1,168 | 1,121 | 47 | 30 | 2.0 / 64 | 0.015 |
| **Val** | `hangouts_chat_4a.pcap` | pcap | Non-VPN | 2.98 | 5,675 | 5,669 | 6 | 108 | 3.0 / 64 | 0.063 |
| **Val** | `ICQchat1.pcapng` | pcapng | Non-VPN | 0.06 | 487 | 465 | 22 | 43 | 2.0 / 64 | 0.008 |
| **Val** | `vpn_hangouts_audio1.pcap` | pcap | VPN | 68.54 | 446,669 | 408,300 | 38,369 | 2,413 | 2.0 / 64 | 4.332 |
| **Test** | `email1a.pcap` | pcap | Non-VPN | 3.06 | 30,410 | 14,354 | 16,056 | 2,857 | 2.0 / 64 | 0.505 |
| **Test** | `facebook_video2b.pcapng`| pcapng | Non-VPN | 223.82 | 245,777 | 245,277 | 500 | 33 | 24.0 / 64 | 3.112 |
| **Test** | `vpn_bittorrent.pcap` | pcap | VPN | 358.76 | 422,098 | 380,921 | 41,177 | 406 | 11.0 / 64 | 6.332 |
| **Total**| **9 Captures** | — | — | **658.18 MB** | **1,153,931** | **1,057,716** | **96,215** | **5,961** | — | **14.44 s** |

### 3.2 Packet Rejection Analysis
- **Accepted Packet Ratio**: 91.66% ($1,057,716$ packets).
- **Rejected Packet Ratio**: 8.34% ($96,215$ packets).
  - *Non-IP Packets*: 15,746 (16.4% of rejected; primarily link-layer broadcasts and ARP in `email1a.pcap`).
  - *Non-Transport Packets*: 80,469 (83.6% of rejected; primarily IPsec ESP encapsulation and ICMP messages in VPN captures).
- **Extraction Runtime & Throughput**:
  - Measured throughput: **45.57 MB/s** on external media.
  - Estimated full-dataset runtime (25.78 GB): **~9.66 minutes** single-worker, **~2.5 to 3 minutes** with 4 parallel worker processes.

---

## 4. Deep Output NPZ Validation

All generated pilot NPZ files (`train_pilot.npz`, `val_pilot.npz`, `test_pilot.npz`) were loaded from disk and rigorously inspected against protocol specifications:

| Check | Requirement | Train NPZ | Val NPZ | Test NPZ | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tensor Shape** | $(N, 2, 64)$ | $(101, 2, 64)$ | $(2564, 2, 64)$ | $(3296, 2, 64)$ | **PASSED** |
| **Data Type** | `float32` | `float32` | `float32` | `float32` | **PASSED** |
| **Missing/NaN/Inf** | 0 NaN, 0 Inf | None | None | None | **PASSED** |
| **Channel 1 Range** | $[-1.0, 1.0]$ | $[-0.927, 0.927]$ | $[-0.927, 0.927]$ | $[-1.000, 1.000]$ | **PASSED** |
| **Channel 2 Range** | $[0, \ln(6)] \approx [0, 1.7918]$ | $[0.0, 1.7918]$ | $[0.0, 1.7918]$ | $[0.0, 1.7918]$ | **PASSED** |
| **Mask Sum vs SeqLen** | $\sum_t M_t = L$ | 100% Match | 100% Match | 100% Match | **PASSED** |
| **Padding Strict Zero**| $X_{c, t} = 0 \;\forall t \ge L$ | 100% Strict Zero | 100% Strict Zero | 100% Strict Zero | **PASSED** |
| **Label Alignment** | $y \in \{0, 1\}$ | Classes $\{0, 1\}$ | Classes $\{0, 1\}$ | Classes $\{0, 1\}$ | **PASSED** |
| **Traceability** | Unique Flow & Capture ID | 100% Valid Strings | 100% Valid Strings | 100% Valid Strings | **PASSED** |

---

## 5. Issue Classification and Gate Verdict

### 5.1 Blocking Issues
* **None**. No scientific or implementation blockers exist.

### 5.2 Non-Blocking Limitations
1. **Limited VPN Recording Groups ($N=20$)**:
   - *Nature*: Structural constraint of the ISCX 2016 dataset.
   - *Impact*: Test set contains 3 VPN capture groups (16 groups total).
   - *Mitigation*: Report both flow-level paired bootstrap and cluster-level bootstrap confidence intervals, explicitly annotating the cluster count limitation in all tables and figures.
2. **Rejection of Non-TCP/UDP Traffic**:
   - *Nature*: Expected behavior per Section 4.1.
   - *Impact*: IPsec ESP and ICMP packets are excluded from 5-tuple flow aggregation.
3. **Environment Warning Suppression**:
   - *Nature*: NumPy 2.0.2 in Anaconda base environment prints deprecation warnings for optional pandas C-extensions (`pyarrow`, `bottleneck`).
   - *Impact*: Handled by fallback; clean execution verified with 14 passing automated unit tests.

### 5.3 Verdict
**VALIDATION GATE: PASSED.**
Authorization is granted to proceed with full-dataset preprocessing and baseline FP32 model training.
