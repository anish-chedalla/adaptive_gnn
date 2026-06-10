# Adaptive GNN Decoding for QLDPC Codes Under Biased + Drifting Noise

> **Accepted poster — QEC 2026 (8th International Quantum Error Correction Conference)**  
> Santa Barbara, CA · June 7–12, 2026 · Board #37 · Submission #195  
> Sponsored by Google Quantum AI

---

## Authors

**Jothiradithya (Sai) Konduru** · sai.konduru10@gmail.com · Paradise Valley High School  
**Anish Chedalla** · anishchedalla@gmail.com · Paradise Valley High School  
**Dr. Nithin Raveendran** · nithin@arizona.edu · UArizona STAR Lab / QEC Labs  

---

## Overview

Quantum computers require decoders that are simultaneously **fast**, **noise-aware**, and **drift-adaptive** — no existing solution satisfies all three requirements. This work presents a novel **GNN-FiLM Assisted Belief Propagation** decoder for Quantum Low-Density Parity-Check (QLDPC) codes that:

- Matches the accuracy of O(n³) BP-OSD at **O(n) complexity** in 4 of 5 noise conditions on [[72,12,6]]
- Achieves **13% logical error rate (LER) reduction** on [[144,12,12]] under unknown Z-bias (McNemar p = 0.026)
- Recovers **25% of the oracle improvement gap** under sinusoidal drift with **zero recalibration**
- Scales with code size: 2–4% improvement at n=72, 9–13% at n=144, up to **60% at n=288**

---

## Key Contributions

### 1. First Interleaved GNN-BP Architecture for Any Quantum Code Family
Prior neural decoders (Maan & Paler, 2025) apply a single correction before decoding. Our architecture intervenes **mid-decoding** using BP's own intermediate marginals to identify and correct trapping set failures in real time.

### 2. First FiLM Conditioning Applied to QLDPC Codes
We repurpose Feature-wise Linear Modulation (FiLM) from visual QA to enable **zero-recalibration drift adaptation**. Syndrome density σ_s = (1/m)Σs_j acts as a real-time noise proxy — no noise labels, no model assumptions, no retraining required.

**FiLM modulation at layer l:**
```
h_l' = γ_l ⊙ h_l + β_l
```
where a 2-layer MLP maps σ_s → (γ_l, β_l) at each timestep.

### 3. End-to-End Differentiable Pipeline
GNN + FiLM + Neural BP trained jointly through differentiable belief propagation. Gradients flow through every BP iteration back into GNN weights, providing per-qubit, per-iteration learning signal.

---

## Architecture

```
Syndrome Input
      │
  TannerGNN  ──── FiLM Conditioning (σ_s → γ_l, β_l)
      │
  Stage 1: GNN correction → Neural BP (K=25 iters)
      │
  Stage 2: GNN mid-correction on BP marginals → Neural BP (K=25 iters)
      │
  Readout MLP → δ_i (per-qubit LLR correction)
      │
  Hard Decision → Logical Error Check
```

**Node features (4-dim):**
- Data qubit: `[LLR_i, 1, 0, 0]`
- X-check: `[s_j, 0, 1, 0]`
- Z-check: `[s_k, 0, 0, 1]`

**Per-edge-type MLPs** for H_X and H_Z edges separately (CSS independence preserved).

---

## Codes Evaluated

| Code | Parameters | Rate |
|------|-----------|------|
| Bivariate Bicycle | [[72, 12, 6]] | k/n = 1/6 |
| Bivariate Bicycle | [[144, 12, 12]] | k/n = 1/6 |
| Bivariate Bicycle | [[288, 12, 18]] | k/n = 1/6 |

All codes constructed from circulant polynomial generators A(x,y), B(x,y) over Z_l × Z_m (Bravyi et al., Nature 2024).

---

## Experimental Results

### Key Findings (McNemar's Paired Binary Test)

| Finding | Ours vs Baseline | Result | p-value |
|---------|-----------------|--------|---------|
| LER reduction under unknown Z-bias | GNN-BP (0.174) vs BP (0.200) | **13% ↓** on [[144,12,12]] | 0.026* |
| O(n) beats O(n³) on OU drift | GNN+BP-LSD (269) vs BP-OSD (275) | O(n) > O(n³) | 0.017* |
| O(n) matches O(n³) accuracy | GNN-BP vs BP-OSD | No sig. diff. 4/5 conditions | >0.05 |
| Consistent improvement across sizes | GNN-BP vs BP | Significant in 13/15 conditions | <0.05* |
| FiLM drift recovery | FiLM GNN-BP vs Oracle | **25% oracle gap** recovered | 0.041* |
| Convergence enhancement | GNN-BP vs BP | 93.2% → 96.3% all 3 sizes | >0.05 |

### Noise Conditions Tested
1. Oracle (true η=20, upper bound)
2. Unknown Bias (decoder assumes η=1, true η=20)
3. Unknown Both (wrong η and wrong p)
4. Sinusoidal Drift + Unknown Bias
5. OU Drift + Unknown Bias (stochastic mean-reverting)

**Protocol:** p ∈ {0.02, 0.03, 0.035, 0.04, 0.05} · 2,000 Monte Carlo shots · 135 total decoder-condition pairs

### Decoder Comparison

| Decoder | Complexity | Handles Bias | Tracks Drift | Degeneracy |
|---------|-----------|-------------|-------------|-----------|
| Baseline BP | O(n) | ✗ | ✗ | ✗ |
| BP-OSD | O(n³) | ✗ | ✗ | ✓ |
| MWPM | O(n³) | ✗ | ✗ | ~ |
| Full Neural | O(n) | ✗ | ✗ | ✗ |
| **Ours (GNN-BP)** | **O(n)** | **✓** | **✓** | **✓** |

---

## Setup

```bash
pip install numpy scipy torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
pip install ldpc pymatching stim
pip install -e .
python -c "import torch; print(torch.cuda.is_available())"
bash run_full_pipeline.sh
```

> **Notes:**
> - Large shot counts (50k) are deliberate for research-grade signal; reduce for quick sanity checks.
> - `threshold_sweep` / `ablation` require `gnn_pipeline.evaluate` API to be consistent.

---

## Future Work

- **EdgeWeightGNN:** per-edge, per-iteration weight modulation inside BP (~30K params; w_ch, w_msg, w_ctv)
- **Hardware validation:** IBM 127-qubit Eagle real syndrome data — zero architecture modifications required
- **Circuit-level noise:** Stim DEM, edge_types=1, observable-based loss
- **FPGA deployment:** 154K params, INT8 quantization → µs-scale latency
- **Cross-code transfer:** freeze message-passing layers, fine-tune [[72,12,6]] → [[288,12,18]]

---

## Citation

If you use this code, please cite:

```bibtex
@misc{konduru2026adaptive,
  title     = {Adaptive Decoding of Quantum LDPC Codes for Realistic Noise Models},
  author    = {Konduru, Jothiradithya and Chedalla, Anish and Raveendran, Nithin},
  year      = {2026},
  note      = {Poster presented at QEC 2026 (8th International Quantum Error Correction Conference), Santa Barbara, CA. Submission \#195.}
}
```

---

## Acknowledgments

We thank Dr. Nithin Raveendran and the UArizona STAR Lab / QEC Labs for their guidance and support.  
This work was presented at QEC 2026, sponsored by Google Quantum AI.

---

## Contact

**Jothiradithya (Sai) Konduru** — sai.konduru10@gmail.com  
**Anish Chedalla** — anishchedalla@gmail.com
