# Related Work

## 1. Encrypted Traffic Representation and Deep Classification
Network traffic analysis has fundamentally evolved as transport-layer encryption (e.g., TLS 1.3, QUIC, WireGuard) rendered payload inspection ineffective for traffic characterization [1], [2]. Consequently, modern Network Intrusion Detection Systems (NIDS) rely on statistical metadata and temporal dynamics [3]. 

Early statistical approaches classified traffic using hand-crafted summary aggregates (e.g., total flow bytes, mean packet durations, and aggregate arrival rates) [4]. However, summary features often fail to capture fine-grained behavioral signatures across protocol state transitions. To overcome this, representation learning approaches directly ingest raw packet sequences [5], [6]. Wang et al. demonstrated that one-dimensional Convolutional Neural Networks (1D-CNNs) can classify encrypted traffic directly from raw byte streams and directional packet lengths [7]. Lotfollahi et al. formalized this with *Deep Packet*, demonstrating high classification fidelity on the ISCX VPN-NonVPN benchmark [8]. Subsequent architectures incorporated multi-channel inputs combining directional packet lengths with inter-arrival times (IAT) [9], recurrent sequence representations [10], semi-supervised graph fingerprinting [11], and pre-trained datagram transformers [12], [13]. Recent investigations in *IEEE Transactions on Information Forensics and Security* have extended these baselines to multi-phase attribute fingerprinting [14] and non-I.I.D. protocol drift [15].

Across these studies, models are evaluated almost exclusively under full 32-bit floating-point (FP32) precision on standard task metrics (accuracy, macro F1, ROC-AUC). While these metrics confirm predictive capability, they do not address how internal feature attributions behave when models are compressed for deployment on perimeter network appliances.

## 2. Fixed-Point Quantization and Edge Inference Constraints
Deploying deep neural networks to perimeter hardware—such as residential gateways, enterprise firewalls, and edge security appliances—is constrained by limited memory bandwidth, cache capacity, and compute budgets [16]. Model compression techniques, notably weight pruning, knowledge distillation, and low-bit quantization, address these constraints [17], [18]. Among these, fixed-point 8-bit integer quantization (INT8) provides substantial throughput and storage benefits on modern edge processors [19], [20].

Quantization strategies generally follow two primary paradigms [21], [22]:
1. **Post-Training Quantization (PTQ):** Linearly maps floating-point dynamic ranges to discrete integer grids using scale ($S$) and zero-point ($Z$) calibration parameters derived from an unlabelled calibration subset [23]. Advanced PTQ formulations optimize integer rounding boundaries via second-order objectives, such as AdaRound [24] and BRECQ [25].
2. **Quantization-Aware Training (QAT):** Models integer rounding during backpropagation using simulated quantization operators and the Straight-Through Estimator (STE) [26], allowing floating-point gradients to update underlying weights through non-differentiable step functions [19].

Low-level inference engines, such as QNNPACK [27] and compiled TorchScript execution graphs, leverage integer SIMD vector instructions to accelerate inference throughput on ARM and x86 edge CPUs. However, quantization benchmarks traditionally focus on preserving predictive accuracy ($\Delta \text{Acc} < 1\%$). The degree to which low-precision integer grids alter internal gradient trajectories and local feature importance rankings remains an active area of investigation.

## 3. Feature Attribution Stability under Model Perturbation and Compression
Post-hoc Explainable AI (XAI) methods are increasingly integrated into security workflows to audit model predictions and identify dominant protocol features [28], [29]. Two primary attribution families dominate operational NIDS literature:
1. **Perturbation-based local surrogates:** Methods such as KernelSHAP [30] and LIME [31] estimate feature importances by sampling local coalitions around an input instance and solving a weighted linear regression, adhering to game-theoretic efficiency and symmetry axioms.
2. **Path-integral gradient methods:** Integrated Gradients (IG) [32] and DeepLIFT [33] compute the integral of gradients along a straight-line interpolation between a baseline reference and the input, satisfying completeness and implementation invariance [34].

Recent evaluations of XAI in network security have applied these methods to examine tabular and deep traffic classifiers [35]–[38]. However, post-hoc explainers exhibit notable sensitivities. Adebayo et al. showed via parameter randomization sanity checks that several saliency methods act independently of learned network weights [39]. Dombrowski et al. proved that local explanation geometry can shift under small input perturbations without altering model outputs [40], and Slack et al. highlighted instability in perturbation-based sampling explainers [41].

Beyond input-level perturbations, model compression introduces structural alterations that can impact feature importance. Hooker et al. demonstrated that model pruning and compression disproportionately degrade performance on rare data subgroups despite stable overall accuracy, showing that compression alters internal representations [42], [43]. Liebenwein et al. similarly observed that pruning alters latent representations beyond what standard test accuracy reveals [44].

More recently, researchers have investigated the direct impact of low-bit quantization on model explanations:
- **Vision Models:** Khandaker introduced *Quant-XAI*, evaluating explanation changes under INT8 PTQ across 2D visual classification architectures using Top-$k$ Intersection-over-Union (IoU) and rank correlation metrics [45]. The study reported that spatial saliency maps can diverge post-quantization even when top-line accuracy is preserved; furthermore, the author evaluated a Quantization-Aware Training (QAT) framework that improved attribution correlation by up to 49.2%, though results varied substantially depending on model architecture (e.g., MobileNetV2 vs. ResNet) [45].
- **Language Models:** Wang et al. investigated uniform quantization in Large Language Models (LLMs) across factual knowledge recall and interpretability tasks [46]. Their findings indicate that while aggressive quantization (e.g., 4-bit) can disrupt latent multi-hop reasoning pathways, degradation is not uniform across model scales, tasks, or quantization algorithms, with certain 8-bit implementations preserving baseline representational stability [46].

## 4. Operational Motivation and Regulatory Context
In Security Operations Centers (SOCs), traffic classifiers often feed automated alert enrichment and Security Orchestration, Automation, and Response (SOAR) workflows [47], [48]. In automated triage pipelines, high-confidence feature attributions may inform security analysts or suggest candidate packet patterns for intrusion detection rule synthesis [49]. If a compressed edge classifier exhibits unstable feature rankings, downstream rules derived from those attributions risk targeting transient or non-causal packet dynamics [50]–[52].

Concurrently, international governance frameworks increasingly emphasize algorithmic transparency and validation in automated systems:
- **General Data Protection Regulation (GDPR):** Article 22 addresses decisions based solely on automated processing that produce legal or similarly significant effects, providing individuals the right to obtain meaningful information about the logic involved [53]. While not all network monitoring tools trigger Article 22, systems performing automated access restriction or user profiling in sensitive environments are subject to heightened scrutiny.
- **EU Artificial Intelligence Act (Regulation EU 2024/1689):** Categorizes specific high-risk AI deployments—including certain critical infrastructure safety components (Annex III)—under strict transparency (Article 13) and human oversight (Article 14) mandates [54]. Deploying compressed classifiers in high-stakes environments requires validating that efficiency optimizations do not introduce uncharacterized behavioral divergences.
- **NIST AI Risk Management Framework (AI RMF 1.0):** Emphasizes that deployed models must remain valid, reliable, and transparent across compression and deployment lifecycles [55].

## 5. Positioning of the Present Study
While previous studies have analyzed quantization-induced attribution shifts in 2D image classifiers [45] and language models [46], the behavior of compressed models on **compact, 1D sequential network metadata** remains under-explored. Network flow representations (e.g., 2-channel tensors of directional packet lengths and log-scaled inter-arrival times) possess unique structural properties: they exhibit strict temporal sequence constraints, variable flow lengths padded with zeros, and extreme class imbalances between background traffic and encrypted tunnels.

The present study, **XDrift**, investigates feature attribution stability when a 1D-CNN encrypted traffic classifier is quantized from FP32 to INT8. Rather than assuming that quantization inherently causes an irreversible collapse of explanations, our experimental design explicitly addresses several methodological gaps:
1. **Evaluating Real Discrete Integer Runtimes:** We benchmark static Post-Training Quantization (PTQ) deployed via genuine integer execution kernels (`torch.qint8` via QNNPACK) on a leakage-free, session-isolated test partition of the ISCX VPN-NonVPN dataset (49,251 test flows across 16 canonical capture groups).
2. **Disentangling Quantization Noise from Optimization Drift:** While prior work noted that QAT alters attribution profiles, standard QAT couples two distinct mechanisms: discrete integer rounding simulation and stochastic gradient descent (SGD) parameter fine-tuning. To isolate these effects, we introduce an **unquantized FP32 fine-tuning control baseline (`FP32-FT`)** that replicates the exact optimizer, learning rate schedule, and loss weighting of QAT without quantization operators. This allows us to determine whether attribution reorganization stems from integer quantization noise or from gradient-driven migration across an overparameterized loss surface.
3. **Characterizing Deployment Artifacts (Quantized Logit Saturation):** We document practical edge-deployment phenomena, such as numerical saturation ceilings in quantized linear output layers that constrain predicted probabilities and necessitate threshold re-calibration.
4. **Benchmarking against Explainer Stochasticity:** We measure empirical attribution shifts against a multi-seed explainer noise baseline using paired non-parametric statistical tests (Wilcoxon signed-rank), establishing whether observed differences reflect genuine model changes or post-hoc sampling variance.

---

## Comparative Taxonomy of Related Work

The following table contextualizes recent literature on model compression and explainability against the experimental scope of this study:

| Study | Modality & Data Type | Compression Scheme | Explainability Method | Attribution Metrics | Includes QAT Evaluation? | Includes Unquantized Fine-Tuning Control? | Session-Isolated Data Splitting? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wang et al. [7]** | 1D Network Traffic (Bytes/Lengths) | None (FP32) | None | N/A | No | No | No (Random Flow Split) |
| **Lotfollahi et al. [8]** | 1D Network Traffic (Packets) | None (FP32) | None | N/A | No | No | No (Random Flow Split) |
| **Aceto et al. [9]** | Multimodal Traffic (Lengths + IAT) | None (FP32) | None | N/A | No | No | Session-level |
| **Sarhan et al. [36]** | Tabular NIDS Features | None (FP32) | SHAP, LIME | Mean Absolute Attribution | No | No | N/A (Standard Splits) |
| **Mali et al. [37]** | Encrypted Traffic Metadata | None (FP32) | SHAP, LIME | Protocol Feature Importance | No | No | Random Flow Split |
| **Hooker et al. [42]** | 2D Images | Pruning & Quantization | None (Subgroup Accuracy) | Class Recall Drop | No | No | Standard ImageNet/CIFAR |
| **Liebenwein et al. [44]** | 2D Images | Unstructured Pruning | Layer-wise Representation | Alignment & Invariance | No | No | Standard ImageNet/CIFAR |
| **Khandaker [45] (Quant-XAI)** | 2D Vision (Images) | INT8 PTQ & QAT | Grad-CAM, LIME | Top-$k$ IoU, Dice, Spearman | Yes (Reported correlation gain) | No | Standard ImageNet/PASCAL |
| **Wang et al. [46]** | NLP (Tokens / Text) | INT8 / INT4 PTQ | Layer Attributions / Probing | Recall Drift, Perplexity | No | No | Standard NLP Benchmarks |
| **XDrift (This Study)** | **1D Flow Metadata (Lengths + IAT)** | **INT8 PTQ & INT8 QAT** | **Masked KernelSHAP** | **Top-$k$ IoU, TV Dist, Signed Cosine** | **Yes (Evaluated on 1D Flows)** | **Yes (`FP32-FT` Control Baseline)** | **Yes (106 Canonical Session Groups)** |

---

## References

1. P. Velan, M. Čermák, P. Čeleda, and M. Drašar, "A survey of methods for encrypted traffic classification and analysis," *International Journal of Network Management*, vol. 25, no. 5, pp. 355–374, 2015.
2. F. Pacheco, E. Exposito, M. Gineste, C. Baudoin, and J. Aguilar, "Towards the deployment of machine learning solutions in network traffic classification: A systematic survey," *IEEE Communications Surveys & Tutorials*, vol. 21, no. 2, pp. 1988–2014, 2019.
3. S. Rezaei and X. Liu, "Deep learning for encrypted traffic classification: An overview," *IEEE Communications Magazine*, vol. 57, no. 5, pp. 76–81, 2019.
4. G. Draper-Gil, A. H. Lashkari, M. S. I. Mamun, and A. A. Ghorbani, "Characterization of encrypted traffic through flow interaction," in *Proc. 2nd Int. Conf. Inf. Syst. Secur. Privacy (ICISSP)*, 2016, pp. 297–304.
5. W. Wang, M. Zhu, X. Zeng, X. Ye, and Y. Sheng, "Malware traffic classification using convolutional neural network for representation learning," in *Proc. IEEE 37th Int. Conf. Distrib. Comput. Syst. (ICDCS)*, 2017, pp. 2061–2067.
6. G. Aceto, D. Ciuonzo, A. Montieri, and A. Pescapè, "Mobile encrypted traffic classification using deep learning," in *Proc. Netw. Traffic Meas. Anal. Conf. (TMA)*, 2018, pp. 1–8.
7. W. Wang, M. Zhu, J. Wang, X. Zeng, and Z. Yang, "End-to-end encrypted traffic classification with one-dimensional convolution neural networks," in *Proc. IEEE Int. Conf. Intell. Secur. Informatics (ISI)*, 2017, pp. 43–48.
8. M. Lotfollahi, M. J. Siavoshani, R. S. H. Zadeh, and M. Saberian, "Deep Packet: A novel approach for encrypted traffic classification using deep learning," *Soft Computing*, vol. 24, no. 3, pp. 1999–2012, 2020.
9. G. Aceto, D. Ciuonzo, A. Montieri, and A. Pescapè, "MIMETIC: Mobile encrypted traffic classification using multimodal deep learning," *Computer Networks*, vol. 165, p. 106944, 2019.
10. C. Liu, L. He, G. Xiong, Z. Cao, and Z. Li, "FS-Net: A flow sequence network for encrypted traffic classification," in *Proc. IEEE INFOCOM*, 2019, pp. 1171–1179.
11. T. van Ede, R. Bortolameotti, A. Continella, J. Ren, D. J. Dubois, M. Lindorfer, D. Choffnes, M. van Steen, and A. Peter, "FlowPrint: Semi-supervised mobile-app fingerprinting on encrypted network traffic," in *Proc. Netw. Distrib. Syst. Secur. Symp. (NDSS)*, 2020.
12. X. Lin, G. Xiong, G. Gou, Z. Li, J. Shi, and J. Yu, "ET-BERT: A contextualized datagram representation with pre-training transformers for encrypted traffic classification," in *Proc. ACM Web Conf. (WWW)*, 2022, pp. 633–642.
13. Q. Zhou et al., "WF-Transformer: Learning temporal features for accurate anonymous traffic identification by using transformer networks," *IEEE Transactions on Information Forensics and Security*, vol. 19, pp. 2736–2751, 2024.
14. Y. Chen et al., "MPAF: Encrypted traffic classification with multi-phase attribute fingerprint," *IEEE Transactions on Information Forensics and Security*, vol. 19, pp. 7091–7105, 2024.
15. X. Lin et al., "Respond to change with constancy: Instruction-tuning with LLM for non-I.I.D. network traffic classification," *IEEE Transactions on Information Forensics and Security*, vol. 20, pp. 5758–5773, 2025.
16. A. Alwarafy, K. A. Al-Thelaya, M. Abdallah, J. Schneider, and M. Hamdi, "A survey on security and privacy issues in edge-computing-assisted Internet of Things," *IEEE Internet of Things Journal*, vol. 8, no. 6, pp. 4004–4022, 2021.
17. S. Han, H. Mao, and W. J. Dally, "Deep Compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2016.
18. T. Liang, J. Glossner, L. Wang, S. Shi, and X. Zhang, "Pruning and quantization for deep neural network acceleration: A survey," *Neurocomputing*, vol. 461, pp. 370–403, 2021.
19. B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 2704–2713.
20. R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper," *arXiv preprint arXiv:1806.08342*, 2018.
21. A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, "A survey of quantization methods for efficient neural network inference," in *Low-Power Computer Vision*, Chapman & Hall/CRC, 2022, pp. 291–326.
22. Y. Bengio, N. Léonard, and A. Courville, "Estimating or propagating gradients through stochastic neurons for conditional computation," *arXiv preprint arXiv:1308.3432*, 2013.
23. M. Nagel, M. van Baalen, T. Blankevoort, and M. Welling, "Data-free quantization through weight equalization and bias correction," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1325–1334.
24. M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos, and T. Blankevoort, "Up or down? Adaptive rounding for post-training quantization," in *Proc. 37th Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 7197–7206.
25. Y. Li, R. Shen, J. Lyu, S. Dong, X. Sheng, and B. Zhang, "BRECQ: Pushing the limit of post-training quantization by block reconstruction," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.
26. M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio, "Binarized neural networks: Training deep neural networks with weights and activations constrained to +1 or -1," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 29, 2016, pp. 561–569.
27. M. Dukhan, "The indirect convolution algorithm," *arXiv preprint arXiv:1907.02129*, 2019.
28. E. Tjoa and C. Guan, "A survey on explainable artificial intelligence (XAI): Toward medical and cyber-security applications," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 32, no. 11, pp. 4793–4813, 2021.
29. Z. Zhang, H. Al Hamadi, E. Damiani, C. Y. Yeun, and F. Taher, "Explainable artificial intelligence applications in cyber security: State-of-the-art in research," *IEEE Access*, vol. 10, pp. 93104–93139, 2022.
30. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 4765–4774.
31. M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why should I trust you?": Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 1135–1144.
32. M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic attribution for deep networks," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 3319–3328.
33. A. Shrikumar, P. Greenside, and A. Kundaje, "Learning important features through propagating activation differences," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 3145–3153.
34. N. Kokhlikyan et al., "Captum: A unified and generic model interpretability library for PyTorch," *arXiv preprint arXiv:2009.07896*, 2020.
35. S. Mane and D. Rao, "Explaining network intrusion detection decisions using explainable AI techniques," *Journal of Computer Virology and Hacking Techniques*, vol. 17, no. 4, pp. 325–336, 2021.
36. M. Sarhan, S. Layeghy, and M. Portmann, "Evaluating the explainability of machine learning models for network intrusion detection systems," *IEEE Transactions on Network and Service Management*, vol. 20, no. 3, pp. 3496–3509, 2023.
37. S. Mali, S. Agarwal, R. Chintawar, and A. K. Cherukuri, "Analyzing encrypted traffic analysis using explainable AI," *Annals of Mathematics and Computer Science*, vol. 31, pp. 1–22, 2025.
38. S. Neupane et al., "Explainable AI for network intrusion detection systems: A comparative study," *IEEE Transactions on Network and Service Management*, vol. 21, no. 1, pp. 842–856, 2024.
39. J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity checks for saliency maps," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, 2018, pp. 9505–9515.
40. A.-K. Dombrowski, M. Alber, C. Anders, M. Ackermann, K.-R. Müller, and P. Kessel, "Explanations can be manipulated and geometry is to blame," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 13589–13600.
41. D. Slack, S. Hilgard, E. Jia, S. Singh, and H. Lakkaraju, "Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods," in *Proc. AAAI/ACM Conf. AI, Ethics, Soc. (AIES)*, 2020, pp. 180–186.
42. S. Hooker, A. Courville, G. Clark, Y. Dauphin, and A. Frome, "What do compressed deep neural networks forget?," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 1464–1475.
43. S. Hooker, N. Moorosi, G. Clark, S. Bengio, and A. Courville, "Characterising bias in compressed models," *arXiv preprint arXiv:2010.03058*, 2020.
44. L. Liebenwein, C. Baykal, B. Carter, D. Gifford, and D. Rus, "Lost in pruning: The effects of pruning neural networks beyond test accuracy," in *Proc. Mach. Learn. Syst. (MLSys)*, vol. 3, 2021, pp. 624–638.
45. S. Khandaker, "Quant-XAI: Quantifying and mitigating explainability drift under INT8 quantization in edge-deployed visual classification," *Research Square Preprint*, 2026, doi: 10.21203/rs.3.rs-9179249/v1.
46. Q. Wang, M. Wang, N. Feldhus, S. Ostermann, Y. Cao, H. Schütze, S. Möller, and V. Schmitt, "Through a compressed lens: Investigating the impact of quantization on factual knowledge recall," in *Proc. 6th Workshop on Trustworthy NLP (TrustNLP 2026) @ ACL*, 2026.
47. C. Islam, M. A. Babar, and S. Nepal, "A multi-vocal review of security orchestration, automation, and response (SOAR)," *ACM Computing Surveys*, vol. 52, no. 2, pp. 1–35, 2019.
48. R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," in *Proc. IEEE Symp. Secur. Privacy (S&P)*, 2010, pp. 305–316.
49. A. Sharma, "XAI-driven encrypted traffic detection and characterization to enhance information security," M.A. thesis, Dept. Elect. Eng. Comput. Sci., York Univ., Toronto, Canada, 2025.
50. M. Usama, J. Qadir, and A. V. Vasilakos, "Adversarial machine learning in network intrusion detection systems: A comprehensive survey," *IEEE Communications Surveys & Tutorials*, vol. 21, no. 2, pp. 1954–1987, 2019.
51. Z. Wang et al., "Adversarial attacks on network intrusion detection systems: A survey," *IEEE Transactions on Network and Service Management*, vol. 18, no. 4, pp. 4104–4124, 2021.
52. Y. Hou et al., "Evading deep learning-based encrypted traffic classification via flow timing and packet size manipulation," *IEEE Transactions on Information Forensics and Security*, vol. 18, pp. 5231–5245, 2023.
53. European Parliament and Council of the European Union, "Regulation (EU) 2016/679 (General Data Protection Regulation)," *Official Journal of the European Union*, L 119, pp. 1–88, 2016.
54. European Parliament and Council of the European Union, "Regulation (EU) 2024/1689 (Artificial Intelligence Act)," *Official Journal of the European Union*, L 2024/1689, 2024.
55. National Institute of Standards and Technology, "Artificial Intelligence Risk Management Framework (AI RMF 1.0)," *NIST AI 100-1*, Gaithersburg, MD, 2023.

---

## Complete BibTeX Entries

```bibtex
@article{velan2015survey,
  author    = {Petr Velan and Milan {\v{C}}erm{\'a}k and Pavel {\v{C}}eleda and Martin Dra{\v{s}}ar},
  title     = {A survey of methods for encrypted traffic classification and analysis},
  journal   = {International Journal of Network Management},
  volume    = {25},
  number    = {5},
  pages     = {355--374},
  year      = {2015},
  doi       = {10.1002/nem.1901}
}

@article{pacheco2019towards,
  author    = {Fabricio Pacheco and Ernesto Exposito and Mathieu Gineste and C{\'e}dric Baudoin and Jose Aguilar},
  title     = {Towards the Deployment of Machine Learning Solutions in Network Traffic Classification: A Systematic Survey},
  journal   = {IEEE Communications Surveys \& Tutorials},
  volume    = {21},
  number    = {2},
  pages     = {1988--2014},
  year      = {2019},
  doi       = {10.1109/COMST.2018.2883147}
}

@article{rezaei2019deep,
  author    = {Shahbaz Rezaei and Xu Liu},
  title     = {Deep Learning for Encrypted Traffic Classification: An Overview},
  journal   = {IEEE Communications Magazine},
  volume    = {57},
  number    = {5},
  pages     = {76--81},
  year      = {2019},
  doi       = {10.1109/MCOM.2019.1800819}
}

@inproceedings{drapergil2016characterization,
  author    = {Gerard Draper-Gil and Arash Habibi Lashkari and Mohammad Saiful Islam Mamun and Ali A. Ghorbani},
  title     = {Characterization of Encrypted Traffic Through Flow Interaction},
  booktitle = {Proceedings of the 2nd International Conference on Information Systems Security and Privacy (ICISSP 2016)},
  pages     = {297--304},
  year      = {2016},
  doi       = {10.5220/0005604102970304}
}

@inproceedings{wang2017endtoend,
  author    = {Wei Wang and Ming Zhu and Jinlin Wang and Xuewen Zeng and Zhongzhen Yang},
  title     = {End-to-end encrypted traffic classification with one-dimensional convolution neural networks},
  booktitle = {Proceedings of the IEEE International Conference on Intelligence and Security Informatics (ISI 2017)},
  pages     = {43--48},
  year      = {2017},
  doi       = {10.1109/ISI.2017.8004872}
}

@article{lotfollahi2020deep,
  author    = {Mohammad Lotfollahi and Mahdi Jafari Siavoshani and Ramin Shirali Hossein Zadeh and Mohammdsadegh Saberian},
  title     = {Deep Packet: A Novel Approach for Encrypted Traffic Classification Using Deep Learning},
  journal   = {Soft Computing},
  volume    = {24},
  number    = {3},
  pages     = {1999--2012},
  year      = {2020},
  doi       = {10.1007/s00500-019-04030-2}
}

@article{aceto2019mimetic,
  author    = {Giuseppe Aceto and Domenico Ciuonzo and Antonio Montieri and Antonio Pescap{\`e}},
  title     = {{MIMETIC}: Mobile encrypted traffic classification using multimodal deep learning},
  journal   = {Computer Networks},
  volume    = {165},
  pages     = {106944},
  year      = {2019},
  doi       = {10.1016/j.comnet.2019.106944}
}

@inproceedings{jacob2018quantization,
  author    = {Benoit Jacob and Skirmantas Kligys and Bo Chen and Menglong Zhu and Matthew Tang and Andrew Howard and Hartwig Adam and Dmitry Kalenichenko},
  title     = {Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018)},
  pages     = {2704--2713},
  year      = {2018},
  doi       = {10.1109/CVPR.2018.00286}
}

@incollection{gholami2022survey,
  author    = {Amir Gholami and Sehoon Kim and Zhen Dong and Zhewei Yao and Michael W. Mahoney and Kurt Keutzer},
  title     = {A Survey of Quantization Methods for Efficient Neural Network Inference},
  booktitle = {Low-Power Computer Vision},
  publisher = {Chapman and Hall/CRC},
  pages     = {291--326},
  year      = {2022}
}

@inproceedings{lundberg2017unified,
  author    = {Scott M. Lundberg and Su-In Lee},
  title     = {A Unified Approach to Interpreting Model Predictions},
  booktitle = {Advances in Neural Information Processing Systems 30 (NeurIPS 2017)},
  pages     = {4765--4774},
  year      = {2017}
}

@inproceedings{sundararajan2017axiomatic,
  author    = {Mukund Sundararajan and Ankur Taly and Qiqi Yan},
  title     = {Axiomatic Attribution for Deep Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning (ICML 2017)},
  pages     = {3319--3328},
  year      = {2017}
}

@article{sarhan2023evaluating,
  author    = {Mohanad Sarhan and Siamak Layeghy and Marius Portmann},
  title     = {Evaluating the Explainability of Machine Learning Models for Network Intrusion Detection Systems},
  journal   = {IEEE Transactions on Network and Service Management},
  volume    = {20},
  number    = {3},
  pages     = {3496--3509},
  year      = {2023},
  doi       = {10.1109/TNSM.2023.3235304}
}

@inproceedings{hooker2019what,
  author    = {Sara Hooker and Aaron Courville and Gregory Clark and Yann Dauphin and Andrea Frome},
  title     = {What Do Compressed Deep Neural Networks Forget?},
  booktitle = {Advances in Neural Information Processing Systems 32 (NeurIPS 2019)},
  pages     = {1464--1475},
  year      = {2019}
}

@inproceedings{liebenwein2021lost,
  author    = {Lucas Liebenwein and Cenk Baykal and Brandon Carter and David Gifford and Daniela Rus},
  title     = {Lost in Pruning: The Effects of Pruning Neural Networks beyond Test Accuracy},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys 2021)},
  volume    = {3},
  pages     = {624--638},
  year      = {2021}
}

@article{khandaker2026quantxai,
  author    = {Sunzil Khandaker},
  title     = {Quant-{XAI}: Quantifying and Mitigating Explainability Drift under {INT8} Quantization in Edge-Deployed Visual Classification},
  journal   = {Research Square Preprint},
  year      = {2026},
  doi       = {10.21203/rs.3.rs-9179249/v1}
}

@inproceedings{wang2026compressed,
  author    = {Qianli Wang and Mingyang Wang and Nils Feldhus and Simon Ostermann and Yuan Cao and Hinrich Sch{\"u}tze and Sebastian M{\"o}ller and Vera Schmitt},
  title     = {Through a Compressed Lens: Investigating the Impact of Quantization on Factual Knowledge Recall},
  booktitle = {Proceedings of the 6th Workshop on Trustworthy NLP (TrustNLP 2026) @ ACL},
  year      = {2026}
}
```
