# QLDPC GNN-BP Pipeline -- Concept Guide

A detailed walkthrough of every major concept in the pipeline,
with ASCII diagrams so you can see how the pieces fit together.


---
## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [CSS Codes & Bivariate Bicycle Codes](#2-css-codes--bivariate-bicycle-codes)
3. [The Tanner Graph](#3-the-tanner-graph)
4. [Belief Propagation (BP) -- The Core Decoder](#4-belief-propagation-bp----the-core-decoder)
5. [Neural BP -- Learnable Weights Per Iteration](#5-neural-bp----learnable-weights-per-iteration)
6. [The GNN (TannerGNN) -- Learning Corrections](#6-the-gnn-tannergnn----learning-corrections)
7. [Correction Modes -- How GNN Fixes LLRs](#7-correction-modes----how-gnn-fixes-llrs)
8. [FiLM Conditioning -- Noise-Adaptive GNN](#8-film-conditioning----noise-adaptive-gnn)
9. [Interleaved GNN-BP -- The Full Pipeline](#9-interleaved-gnn-bp----the-full-pipeline)
10. [All Decoder Types Compared](#10-all-decoder-types-compared)
11. [Loss Functions -- What We Optimize](#11-loss-functions----what-we-optimize)
12. [Drift Models -- Time-Varying Noise](#12-drift-models----time-varying-noise)
13. [Online Data Regeneration](#13-online-data-regeneration)
14. [The 7 Publication Figures](#14-the-7-publication-figures)
15. [Circuit-Level Decoding (DEM)](#15-circuit-level-decoding-dem)
16. [Quick Reference: Pipeline Commands](#16-quick-reference-pipeline-commands)

---

## 1. The Big Picture

The pipeline takes a quantum error-correcting code, simulates noisy
quantum operations, and uses a Graph Neural Network to help a classical
decoder (Belief Propagation) correct the errors.

```
                          THE FULL PIPELINE
 ============================================================

  QUANTUM HARDWARE          CLASSICAL PROCESSING
  (simulated by Stim)       (our pipeline)

  +-----------+    syndromes    +--------+   corrected   +--------+
  |  Physical |  +  channel  -> |  GNN   | ->  LLRs   -> |  BP    | -> decoded
  |  Qubits   |    LLRs        | (learn)|               |(decode)|    errors
  +-----------+                 +--------+               +--------+
       |                            ^                        |
       | errors                     |                        v
       v                      Tanner graph             logical
  noise model                 (code structure)         error check
  (p, eta, drift)
```

**The key insight**: Standard BP struggles on QLDPC codes because of
short cycles in the Tanner graph. The GNN learns to "pre-correct" the
channel information so BP converges better.


---

## 2. CSS Codes & Bivariate Bicycle Codes

### What is a CSS Code?

A CSS (Calderbank-Shor-Steane) code is a quantum code where X and Z
error correction are handled independently using two classical parity-
check matrices:

```
  CSS Code Structure
  ==================

  n physical qubits encodes k logical qubits

  Hx  (mx x n)  -- detects Z errors (X-type stabilizers)
  Hz  (mz x n)  -- detects X errors (Z-type stabilizers)

  Key property:  Hx @ Hz^T = 0  (mod 2)
                 (X and Z checks are compatible)


  DECODING IS SEPARATE:
  +---------------------------------------------------+
  |                                                   |
  |   Z errors:  x_syndrome = Hx @ z_error  (mod 2)  |
  |   X errors:  z_syndrome = Hz @ x_error  (mod 2)  |
  |                                                   |
  |   NEVER combine into one big matrix!              |
  |   np.vstack([Hx, Hz]) DOES NOT WORK for CSS.     |
  |                                                   |
  +---------------------------------------------------+
```

### Bivariate Bicycle Codes

Our codes come from the bivariate bicycle construction (Bravyi et al.):

```
  Code Parameters
  ===============

  [[n, k, d]] = [[block_length, logical_qubits, distance]]

  +---------------+-------+-------+-------+
  | Code          |   n   |   k   |   d   |
  +---------------+-------+-------+-------+
  | [[72,12,6]]   |   72  |   12  |   6   |   <-- our PoC code
  | [[144,12,12]] |  144  |   12  |  12   |   <-- medium
  | [[288,12,18]] |  288  |   12  |  18   |   <-- large
  +---------------+-------+-------+-------+

  All encode k=12 logical qubits.
  Larger n means more redundancy => better protection.
  Distance d means it can correct floor((d-1)/2) errors.
```

### Biased Noise Model

Real quantum hardware has asymmetric noise -- Z errors happen much
more often than X errors:

```
  Biased Noise (eta = 20)
  =======================

  Total error probability:  p = 0.02  (2% per qubit)

  Z error probability:  pz = p * eta / (eta + 1) = 0.02 * 20/21 ~ 0.019
  X error probability:  px = p / (eta + 1)       = 0.02 * 1/21  ~ 0.001

  In log-likelihood ratio (LLR):
    LLR_z = ln((1-pz)/pz) ~ 3.9    (moderate confidence "no error")
    LLR_x = ln((1-px)/px) ~ 6.9    (high confidence "no error")

  Positive LLR = probably no error
  Negative LLR = probably has error
  |LLR| = confidence level
```


---

## 3. The Tanner Graph

The Tanner graph is the key data structure. It represents the code's
parity-check structure as a bipartite graph that the GNN operates on.

```
  Tanner Graph for CSS Code
  =========================

     DATA QUBITS          X-CHECKS           Z-CHECKS
     (type 0)             (type 1)           (type 2)
     n = 72 nodes         mx = 30 nodes      mz = 30 nodes

       q0 ------edge type 0------- Cx0
       q1 ------edge type 0------- Cx1
       q2 ------edge type 0------- Cx0
       q2 ------edge type 1------- Cz0     <-- same qubit, both types
       q3 ------edge type 1------- Cz1
       ...

  BIPARTITE:  edges only between data qubits and checks, never
              check-to-check or qubit-to-qubit

  Total nodes:  n + mx + mz  = 72 + 30 + 30 = 132
  Edges:        2 * (nnz(Hx) + nnz(Hz))     (bidirectional)

  Node features (4-dim):
  +------+----------+----------+----------+----------+
  | Node | feat[0]  | feat[1]  | feat[2]  | feat[3]  |
  +------+----------+----------+----------+----------+
  | Data | chan_LLR  | 1 (flag) | 0        | 0        |
  | X-chk| syndrome | 0        | 1 (flag) | 0        |
  | Z-chk| syndrome | 0        | 0        | 1 (flag) |
  +------+----------+----------+----------+----------+

  This tells the GNN:
    - What type of node am I? (one-hot in features 1-3)
    - What information do I carry? (LLR or syndrome in feature 0)
```

### How It's Built

```python
  build_tanner_graph(hx, hz)
  # Returns: (node_type, edge_index, edge_type)
  #
  # 1. Find all nonzero entries in Hx -> edges of type 0
  # 2. Find all nonzero entries in Hz -> edges of type 1
  # 3. Make bidirectional: data->check AND check->data
  # 4. Assign node types: 0=data, 1=X-check, 2=Z-check
```


---

## 4. Belief Propagation (BP) -- The Core Decoder

BP is a message-passing algorithm on the Tanner graph. It iteratively
refines beliefs about which qubits have errors.

```
  Min-Sum BP (one iteration)
  ==========================

  STEP 1: Variable-to-Check (VTC) messages
  -----------------------------------------

    Each data qubit j sends a message to each check i:

      v_to_c(j->i) = channel_LLR[j] + SUM(c_to_v(k->j))
                                        for all checks k != i

    "My current belief about myself, excluding check i's opinion"


  STEP 2: Check-to-Variable (CTV) messages
  -----------------------------------------

    Each check i sends a message to each data qubit j:

      c_to_v(i->j) = alpha * SIGN * MIN(|v_to_c(k->i)|)
                                      for all data qubits k != j

    alpha ~ 0.75 is the min-sum scaling factor (approximation to sum-product)
    SIGN = product of signs of all other incoming messages
    MIN  = minimum absolute value of all other incoming messages

    "Based on all OTHER qubits connected to me, here's what I think about qubit j"


  STEP 3: Compute beliefs (marginals)
  ------------------------------------

    total_LLR[j] = channel_LLR[j] + SUM(c_to_v(i->j))  for all checks i

    P(error_j) = sigmoid(-total_LLR[j])

    If P > 0.5 => declare error at qubit j


                    iteration 1         iteration 2
                 +---+---------+      +---+---------+
  channel_LLR ->| V |         |  ->  | V |         |  ->  ...  -> final
                | T | C-to-V  |      | T | C-to-V  |             marginals
                | C |         |      | C |         |
                 +---+---------+      +---+---------+

  Typical: 100 iterations, but often converges in 10-30
  Problem: on QLDPC codes, convergence rate is only ~5% at p=0.02!
```

### Convergence Check

```
  After each iteration, check if the current hard decision
  satisfies ALL syndromes:

    hard_decision[j] = 1  if total_LLR[j] < 0  else 0
    predicted_syndrome = H @ hard_decision  (mod 2)

    CONVERGED if predicted_syndrome == actual_syndrome

  Vectorized:  (hard_decision @ H^T) mod 2  == syndrome
```


---

## 5. Neural BP -- Learnable Weights Per Iteration

Standard BP uses fixed weights (all 1.0). Neural BP learns an optimal
weight for each iteration, allowing the algorithm to adapt its
aggressiveness over time.

```
  Neural BP: 3 Learnable Weights Per Iteration
  =============================================

  Standard BP:                    Neural BP:
    VTC = 1.0 * ch + 1.0 * msg     VTC = w_ch[t] * ch + w_vtc[t] * msg
    CTV = alpha * sign * min        CTV = alpha * w_ctv[t] * sign * min

  Where t = iteration number (0, 1, 2, ...)


  WEIGHT PARAMETERIZATION:
  +-------------------------------------------------------------+
  |  Raw parameter: theta  (stored as nn.Parameter, init = 0)   |
  |                                                             |
  |  Bounded weight: w = softplus(theta) / ln(2)               |
  |                                                             |
  |  theta = 0   =>  w = softplus(0)/ln2 = ln(2)/ln(2) = 1.0  |
  |  theta < 0   =>  w < 1.0  (damping)                        |
  |  theta > 0   =>  w > 1.0  (amplification)                  |
  +-------------------------------------------------------------+

  WHAT IT LEARNS:
  +-----------+---------------------------------------------+
  | Iteration | Typical learned behavior                    |
  +-----------+---------------------------------------------+
  |  Early    | w_ctv < 1.0 -- damp check messages to      |
  | (1-3)     | avoid oscillation from short cycles         |
  +-----------+---------------------------------------------+
  |  Middle   | w_ch slightly > 1.0 -- trust channel more  |
  | (4-10)    | as BP might be going off track              |
  +-----------+---------------------------------------------+
  |  Late     | w_vtc, w_ctv ~ 1.0 -- let standard BP      |
  | (10+)     | take over as beliefs stabilize              |
  +-----------+---------------------------------------------+

  Result: ~same convergence rate as vanilla BP, BUT the marginals
  at non-convergence are significantly better quality.
```


---

## 6. The GNN (TannerGNN) -- Learning Corrections

The Graph Neural Network operates on the Tanner graph and outputs
per-qubit corrections to the channel LLRs.

```
  TannerGNN Architecture
  ======================

  Input: node features (num_nodes, 4)
         edge_index, edge_type


                        +------------------+
   node features  ----> | Input Projection |  (Linear 4 -> hidden_dim)
   (n+mx+mz, 4)        | + LayerNorm      |  handles mixed scales
                        +------------------+  (LLR~20 vs indicators~1)
                              |
                              v
                   +----------------------+
                   |  Message Passing x L  |  L = num_mp_layers (3-5)
                   |  (see detail below)   |
                   +----------------------+
                              |
                              v
                   +----------------------+
                   | Readout (data only)   |  Linear hidden_dim -> 1
                   | Only on type-0 nodes  |  (or -> 2 for "both" mode)
                   +----------------------+
                              |
                              v
                   correction per qubit (n,)


  MESSAGE PASSING LAYER (one layer):
  ==================================

  For each edge type separately:

    1. Gather source features:  h_src[edge_index[0]]
    2. Linear transform:        msg = W_type * h_src
    3. Aggregate to target:     agg[dst] += msg   (sum over neighbors)

  Then per node:

    h_new = ReLU(agg)                     basic
    h_new = h + ReLU(agg)                 +residual (use_residual=True)
    h_new = LayerNorm(h + ReLU(agg))      +layer_norm (use_layer_norm=True)


  WITH ATTENTION (use_attention=True):
  ====================================

    1. Concatenate source + dest features:  [h_src || h_dst]
    2. Linear -> 1 scalar -> sigmoid:  score = sigmoid(W * [h_src||h_dst])
    3. Multiply messages by scores:    msg = score * W_type * h_src
    4. Aggregate weighted messages

    This lets the GNN learn WHICH neighbors to listen to.

    Parameter count: ~154K (no attn) vs ~1M (with attn) at hidden=128, L=5
```

### What Does the GNN Learn?

```
  The GNN's job is NOT to decode -- that's BP's job.
  The GNN's job is to FIX the channel LLRs before BP sees them.

  Example of what the GNN might learn:

    Qubit q7 has channel_LLR = +3.0  (weak "no error" belief)
    But q7 is surrounded by checks that ALL have syndrome = 1
    => The GNN outputs correction = -4.5
    => Corrected LLR = 3.0 + (-4.5) = -1.5  (now believes "error"!)

    Qubit q42 has channel_LLR = -0.5  (weak "error" belief)
    But only 1 of 6 connected checks has syndrome = 1
    => The GNN outputs correction = +2.0
    => Corrected LLR = -0.5 + 2.0 = +1.5  (corrected false alarm)

  The GNN sees the GLOBAL structure (via message passing layers)
  and can detect patterns that local BP messages miss.
```


---

## 7. Correction Modes -- How GNN Fixes LLRs

Three ways the GNN output can modify the channel LLR before it enters BP:

```
  ADDITIVE MODE  (correction_mode="additive")
  ============================================

    corrected = channel_LLR + gnn_output

    channel_LLR:   [ 3.0,  -0.5,  5.2,   1.1 ]
    gnn_output:    [-4.5,   2.0, -0.1,  -3.0 ]
                     +       +      +       +
    corrected:     [-1.5,   1.5,   5.1,  -1.9 ]

    Simple bias shift. GNN adds/subtracts from channel belief.


  MULTIPLICATIVE MODE  (correction_mode="multiplicative")
  =======================================================

    corrected = channel_LLR * exp(clamp(gnn_output, -5, 5))

    gnn_output = -1.0  =>  scale factor = exp(-1) = 0.37  (shrink LLR)
    gnn_output =  0.0  =>  scale factor = exp(0)  = 1.00  (no change)
    gnn_output =  1.0  =>  scale factor = exp(1)  = 2.72  (amplify LLR)

    Clamping to [-5, 5] prevents overflow:
      exp(-5) = 0.0067   (almost zero out the LLR)
      exp(+5) = 148.4    (strongly amplify)

    Key advantage: preserves SIGN of LLR (just scales magnitude).
    Good when channel LLR direction is usually right.


  "BOTH" MODE  (correction_mode="both")
  ======================================

    GNN outputs 2 values per qubit: [mul, add]

    corrected = channel_LLR * exp(clamp(mul, -5, 5)) + add

    Most expressive -- can both rescale AND shift.
    Readout layer: hidden_dim -> 2 (instead of -> 1)
```


---

## 8. FiLM Conditioning -- Noise-Adaptive GNN

FiLM (Feature-wise Linear Modulation) lets the GNN adapt its behavior
based on the current noise level. One model works across ALL noise rates.

```
  THE PROBLEM WITHOUT FiLM:
  =========================

    At p = 0.005:  LLR ~ 5.3   (high confidence)
    At p = 0.050:  LLR ~ 2.9   (low confidence)

    The GNN needs DIFFERENT correction strategies at different p!
    Without FiLM, it must learn a compromise that works for all p.


  FiLM: "Tell the GNN what noise level it's working with"
  ========================================================

    Input: p_value (1 scalar per shot in the batch)

    p_value ---> FiLMGenerator MLP ---> per-layer (gamma, beta)
       |
       |    +------------------------------------------+
       +--> | Linear(1, 64) -> ReLU                    |
            | Linear(64, 64) -> ReLU                   |
            | Linear(64, num_layers * 2 * hidden_dim)  |
            +------------------------------------------+
                              |
                    reshape into L pairs of (gamma, beta)
                    each gamma, beta has shape (batch_size, hidden_dim)


  HOW FiLM MODULATES EACH GNN LAYER:
  ====================================

    Normal message passing:   h_new = ReLU(aggregate(messages))

    With FiLM:                h_new = ReLU(gamma * aggregate(messages) + beta)

    Where gamma and beta come from the FiLMGenerator.

    gamma = 1.0, beta = 0.0  =>  identity (no modulation)
    gamma = 0.5, beta = 0.0  =>  dampen all features by half
    gamma = 1.0, beta = 2.0  =>  shift all features up by 2


  BROADCAST FROM BATCH TO NODES:
  ==============================

    batch = [0, 0, 0, ..., 1, 1, 1, ...]   (which sample each node belongs to)

    gamma has shape (B, hidden_dim)  -- one per sample in batch
    gamma[batch] has shape (num_nodes, hidden_dim)  -- expanded to all nodes

    So all nodes from sample #3 get the same gamma/beta,
    which depends on sample #3's noise level.


  INITIALIZATION (Key Detail):
  ============================

    Last layer weights = 0
    Last layer bias = [1, 1, ..., 0, 0, ...]
                       ^^^^^^^^   ^^^^^^^^
                       gamma=1    beta=0

    => At initialization, FiLM does NOTHING (identity).
    => Training gradually learns useful modulations.
    => Prevents FiLM from destabilizing early training.
```

### FiLM Diagram

```
                     p = 0.02
                       |
               +-------v--------+
               |  FiLM Generator |
               |  (small MLP)    |
               +-------+--------+
                       |
           +-----------+-----------+
           |           |           |
      (gamma_1,    (gamma_2,   (gamma_3,
       beta_1)      beta_2)     beta_3)
           |           |           |
           v           v           v
       +--------+  +--------+  +--------+
       | GNN    |  | GNN    |  | GNN    |
       |Layer 1 |  |Layer 2 |  |Layer 3 |
       |        |  |        |  |        |
       | h*g+b  |  | h*g+b  |  | h*g+b  |
       +--------+  +--------+  +--------+
```


---

## 9. Interleaved GNN-BP -- The Full Pipeline

This is the most powerful decoder: it runs Neural BP, then lets the
GNN correct mid-stream, then continues with standard BP.

```
  STANDARD GNN-BP (one-shot correction):
  =======================================

     channel_LLR -> [GNN] -> corrected_LLR -> [BP 100 iters] -> result

     GNN fires ONCE, before BP starts.
     BP never gets a second chance to use GNN insights.


  INTERLEAVED GNN-BP (mid-stream correction):
  ============================================

       Stage 1: Neural BP               Stage 2: Standard BP
      (learned weights)                 (vanilla, 90 iters)
    +------------------------+      +------------------------+
    |                        |      |                        |
    |  channel_LLR           |      |  corrected_LLR         |
    |     |                  |      |     |                  |
    |     v                  |      |     v                  |
    |  Neural BP             |      |  Standard BP           |
    |  10 iterations         | ---> |  ~90 iterations        | -> result
    |  w_ch, w_vtc, w_ctv    |  |   |  fixed weights         |
    |     |                  |  |   |                        |
    |     v                  |  |   +------------------------+
    |  intermediate          |  |
    |  marginals             |  |
    +------------------------+  |
                                |
                           GNN sees the
                           intermediate state
                           and corrects LLRs
                                |
                         +--------------+
                         |  GNN(x_feat) |
                         |  -> delta    |
                         +--------------+
                                |
                         corrected_LLR =
                         channel_LLR + delta


  WHY THIS IS BETTER:
  ====================

  1. Stage 1 (Neural BP) runs with LEARNED damping weights.
     These learned weights prevent early oscillation in QLDPC codes.

  2. After 10 iterations, the GNN can see HOW BP is doing:
     - Which qubits does BP think have errors?
     - Where is BP uncertain (marginals near 0.5)?
     - Which syndromes are not yet satisfied?

  3. The GNN uses this INTERMEDIATE STATE to produce better corrections
     than it could from the raw channel LLR alone.

  4. Stage 2 (Standard BP) gets a "second chance" with corrected LLRs,
     often converging where one-shot GNN-BP would fail.


  IMPLEMENTATION (bp_decoder.py: forward_stages):
  ================================================

    def forward_stages(channel_llr, syndrome, correction_fn, ...):
        for stage in range(num_stages):       # typically 2 stages
            for it in range(iters_per_stage):  # stage1=10, stage2=90
                # ... VTC, CTV message passing ...
                # Uses neural weights if stage==0, vanilla if stage>0

            if stage < num_stages - 1:
                mid_marginals = sigmoid(-total_llr)
                current_llr = correction_fn(mid_marginals, current_llr, stage)
                # ^ GNN fires here, between stages
```


---

## 10. All Decoder Types Compared

```
  DECODER FAMILY TREE
  ===================

  Pure BP family:
  +-- BP (vanilla)           standard min-sum, fixed alpha=0.75
  +-- Oracle BP              BP with TRUE per-shot error rate (cheating upper bound)
  +-- Neural BP              BP with learned per-iteration weights

  GNN-enhanced:
  +-- GNN-BP                 GNN corrects LLRs once, then BP
  +-- Interleaved GNN-BP     Neural BP stage 1, GNN corrects, BP stage 2

  Post-processing combos:
  +-- BP-OSD                 BP + Ordered Statistics Decoding (ldpc library)
  +-- BP-LSD                 BP + Localized Statistics Decoding
  +-- BeliefFind             BP + Union-Find fallback
  +-- GNN + BP-OSD           GNN corrects LLRs, then BP-OSD
  +-- GNN + BP-LSD           GNN corrects LLRs, then BP-LSD

  Matching:
  +-- MWPM                   Minimum Weight Perfect Matching (PyMatching)
```

### How Each Decoder Works

```
  BP (Belief Propagation)
  =======================

     channel_LLR --> [min-sum BP, 100 iters] --> hard decision
                                                      |
     - Simplest, fastest                              v
     - Convergence: ~5% at p=0.02              syndrome check
     - LER: ~10% (high)


  Oracle BP
  =========

     TRUE p_value --> compute exact LLR --> [BP 100 iters] --> hard decision

     - Knows the TRUE per-shot noise rate (impossible in practice)
     - Upper bound on what ANY LLR-based decoder could do
     - Useful as a reference: "if we knew p perfectly, how good would BP be?"


  BP-OSD (BP + Ordered Statistics Decoding)
  =========================================

     channel_LLR --> [BP ~100 iters] -+-> (converged?) --> done
                                      |
                                      +-> (not converged?) --> OSD post-processing
                                                                    |
     OSD: Gaussian elimination on PCM, ordered by BP reliability    |
     Guaranteed to find a valid codeword                            v
                                                               low-weight
                                                               correction

     - Very strong: LER ~ 1-3%
     - Slow: O(n^3) worst case for OSD
     - Industry standard for QLDPC


  BP-LSD (BP + Localized Statistics Decoding)
  ============================================

     channel_LLR --> [BP ~100 iters] --> soft marginals
                                              |
                                              v
                                         local search
                                         (flip bits to reduce
                                          unsatisfied checks)
                                              |
                                              v
                                         LSD solution

     - Similar performance to BP-OSD
     - LSD uses local structure to find corrections
     - lsd_order=0 is simplest (LSD-CS or LSD-0)


  BeliefFind (BP + Union-Find)
  ============================

     channel_LLR --> [BP ~100 iters] -+-> (converged?) --> done
                                      |
                                      +-> (not converged?) --> Union-Find
                                                                   |
     Union-Find: grows clusters around unsatisfied checks          |
     Merges overlapping clusters, flips bits to fix                v
                                                              UF correction

     - Faster than OSD (near-linear time)
     - Slightly worse accuracy


  MWPM (Minimum Weight Perfect Matching)
  =======================================

     syndrome --> build weighted graph --> PyMatching --> correction
                  (edges between checks,
                   weights from LLR)

     - Optimal for surface codes
     - APPROXIMATE for QLDPC (columns have >2 nonzeros)
     - We decompose LDPC columns via chaining + boundary edges
     - LER ~ 28% at p=0.02 (worst decoder -- QLDPC is not surface code!)
     - Included as a reference baseline only


  GNN-BP (GNN-corrected BP)
  =========================

     channel_LLR --+-->  [GNN]  --> correction
                   |                    |
                   +--------------------+---> corrected_LLR --> [BP] --> done

     - GNN fires ONCE before BP
     - LER improvement: ~20-30% vs plain BP


  Interleaved GNN-BP
  ==================

     channel_LLR --> [Neural BP, 10 iters] --> intermediate state
                          |
                          v
                      [GNN correction]
                          |
                          v
                     [Standard BP, 90 iters] --> done

     - Best decoder: ~30-50% LER improvement vs plain BP
     - GNN sees intermediate BP state for smarter corrections


  GNN + BP-LSD / GNN + BP-OSD
  ============================

     channel_LLR --> [GNN] --> corrected_LLR --> [BP-LSD or BP-OSD] --> done

     - GNN pre-correction + strong post-processing
     - Best of both worlds
     - For GNN+BP-LSD: converts GNN-corrected LLRs to per-qubit probs
       via 1/(1+exp(LLR)), passes as error_channel to the LSD decoder
```

### Performance Summary (typical, [[72,12,6]], p=0.02, eta=20)

```
  +------------------------+-------+----------------+
  | Decoder                |  LER  |  vs BP         |
  +------------------------+-------+----------------+
  | MWPM                   | ~28%  |  much worse    |
  | BP                     | ~10%  |  baseline      |
  | Oracle BP              |  ~5%  |  upper bound   |
  | GNN-BP                 |  ~7%  |  ~30% better   |
  | Interleaved GNN-BP     |  ~6%  |  ~40% better   |
  | BP-OSD                 | ~1-3% |  ~80% better   |
  | BP-LSD                 | ~1-3% |  ~80% better   |
  | GNN + BP-LSD           | ~0.5% |  ~95% better   |
  +------------------------+-------+----------------+

  Note: exact numbers depend on training, noise model, etc.
```


---

## 11. Loss Functions -- What We Optimize

The choice of loss function determines WHAT the GNN learns to optimize.

```
  WEIGHTED BCE (Binary Cross-Entropy)
  ====================================

    loss = -[w * target * log(p) + (1-target) * log(1-p)]

    w = pos_weight ~ 50  (because errors are rare: ~2% of qubits)

    Without weighting: model predicts "no error everywhere" (98% accurate but useless)
    With weighting:    penalizes missed errors 50x more than false alarms

    Default for supervised training.


  FOCAL LOSS
  ==========

    loss = -alpha * (1 - p_t)^gamma * log(p_t)

    p_t = p if target=1, else (1-p)

    alpha = 0.25, gamma = 2.0

    The (1 - p_t)^gamma term:
      - If model is CONFIDENT and CORRECT:  (1-0.99)^2 = 0.0001  (tiny loss)
      - If model is UNCERTAIN:              (1-0.50)^2 = 0.25    (moderate loss)
      - If model is CONFIDENT and WRONG:    (1-0.01)^2 = 0.98    (huge loss)

    Automatically focuses training on HARD EXAMPLES.
    Great for imbalanced problems (rare errors).


  SYNDROME CONSISTENCY LOSS (no ground truth needed!)
  ===================================================

    Instead of comparing to TRUE errors, check if predicted errors
    produce the CORRECT syndrome:

    predicted_marginals:  [0.02, 0.95, 0.01, 0.87, ...]
                                                  |
                                                  v
    predicted_syndrome = XOR_probability(marginals along each check row)

    loss = BCE(predicted_syndrome, actual_syndrome)

    Uses exact XOR formula:
      P(odd parity) = 0.5 * (1 - product(1 - 2*p_i))

    Key insight: many different error patterns produce the same syndrome
    (degeneracy), so this loss doesn't penalize valid alternative solutions.


  COSET LOSS (degeneracy-aware)
  =============================

    loss = syn_weight * syndrome_loss + logical_weight * logical_loss

    syndrome_loss:  "do predicted errors match the syndrome?"
    logical_loss:   "do predicted errors flip the same logical qubits?"

    Does NOT penalize different physical error patterns that belong
    to the same error equivalence class (coset).


  OBSERVABLE LOSS (circuit-level, no ground truth)
  ================================================

    For circuit-level decoding, we don't know the TRUE fault pattern.
    We only know:
      1. Which detectors fired (syndrome)
      2. Which logical observables flipped

    loss = obs_weight * observable_flip_loss + syn_weight * syndrome_loss

    observable_flip_loss:  P(predicted errors flip observable) vs actual flip
    syndrome_loss:         P(predicted errors trigger detector) vs actual

    Enables training WITHOUT ground-truth error patterns.


  PER-ITERATION LOSS
  ==================

    loss = SUM over t:  decay^(T-1-t) * base_loss(marginals_at_iter_t)

    decay = 0.8, T = total iterations

    Early iterations:  weight = 0.8^9 = 0.13  (small)
    Last iteration:    weight = 0.8^0 = 1.00  (full)

    Provides gradient signal at EVERY BP iteration, not just the last.
    Helps with vanishing gradients through many BP iterations.
```


---

## 12. Drift Models -- Time-Varying Noise

In real quantum hardware, the error rate changes over time. We simulate
three types of drift:

```
  SINE DRIFT (deterministic)
  ==========================

    p(t) = p_base + amp * sin(2*pi*t / period)

    p_base = 0.02, amp = 0.015, period = 100

         p
    0.035 |         *                       *
          |       *   *                   *   *
    0.020 |-----*-------*-------*-------*-------*-----   <- p_base
          |   *           *   *           *
    0.005 | *               *               *
          +-------------------------------------------> time (shots)
           0        50       100      150      200

    Simple, periodic. Models slow calibration drift.


  ORNSTEIN-UHLENBECK (OU, mean-reverting random walk)
  ====================================================

    dp = -theta * (p - p_base) * dt + sigma * dW

    theta = 0.1  (pull-back strength)
    sigma = 0.005 (randomness)

         p
    0.030 |    *   *
          |  *   *   *         *
    0.020 |-*---------*--*---*---*---*-*-   <- pulled toward p_base
          |             *   *       *   *
    0.010 |                            *
          +-------------------------------------------> time

    Random but bounded. Wanders around p_base, pulled back if too far.
    Most realistic model for gradual environmental drift.


  RANDOM TELEGRAPH NOISE (RTN, Markov switching)
  ===============================================

    Two states:  p_low = p_base - p_delta
                 p_high = p_base + p_delta

    Switch between states with probability switch_prob per shot.

         p
    0.030 |  +-----------+      +---+    +----+
          |  |           |      |   |    |    |
    0.020 |--+           +------+   +----+    +---
          |
    0.010 |
          +-------------------------------------------> time

    Binary jumps. Models two-level system (TLS) defects in qubits.
    p_delta = 0.01, switch_prob = 0.005
```

### Why Drift Matters for the GNN

```
  WITHOUT drift awareness (standard decoder):
    - Decoder assumes p = 0.02 for ALL shots
    - When p drifts to 0.035, LLRs are WRONG (too confident)
    - BP doesn't converge, LER spikes

  WITH FiLM conditioning:
    - GNN knows per-shot p_value (passed as FiLM input)
    - At p = 0.035: GNN learns to dampen LLRs (less confidence)
    - At p = 0.005: GNN learns to amplify LLRs (more confidence)
    - Adapts in real-time, no recalibration needed
```


---

## 13. Online Data Regeneration

Instead of training on a fixed dataset, generate FRESH data every epoch.

```
  FIXED DATASET (traditional):
  ============================

    Epoch 1:  train on sample_1, sample_2, ... sample_5000
    Epoch 2:  train on sample_1, sample_2, ... sample_5000  (SAME data!)
    Epoch 10: train on sample_1, sample_2, ... sample_5000  (SAME data!)

    Risk: GNN memorizes the specific error patterns.


  ONLINE REGENERATION:
  ====================

    Epoch 1:  generate 5000 FRESH samples -> train
    Epoch 2:  generate 5000 FRESH samples -> train (completely new!)
    Epoch 10: generate 5000 FRESH samples -> train (completely new!)

    Every epoch sees brand-new error patterns.
    Infinite effective dataset size.


  HOW IT WORKS (OnlineCodeCapDataset):
  ====================================

    __init__():
      - Store code matrices (Hx, Hz, Lx, Lz)
      - Build Tanner graph ONCE (expensive, reuse)
      - Store noise parameters (p_base, eta, p_range)

    set_epoch(epoch):
      - New RNG seed = base_seed + epoch
      - For each of samples_per_epoch shots:
          1. Sample p ~ Uniform(p_range[0], p_range[1])
          2. Compute pz, px from p and eta
          3. Sample z_errors ~ Bernoulli(pz) for each qubit
          4. Sample x_errors ~ Bernoulli(px) for each qubit
          5. Compute syndromes: x_syn = Hx @ z_err % 2
          6. Compute channel LLR: ln((1-pz)/pz)
      - Store all samples in memory

    __getitem__(idx):
      - Return pre-computed sample as torch_geometric Data object


  p_range FOR CURRICULUM:
  =======================

    p_range = (0.01, 0.04)

    Each shot gets a RANDOM p in [0.01, 0.04]:
      - Some easy (p=0.01, few errors)
      - Some hard (p=0.04, many errors)
      - GNN learns to handle the full range

    Combined with FiLM, the GNN knows which regime each shot is from.
```


---

## 14. The 7 Publication Figures

All generated by `python -m gnn_pipeline.make_figures`

```
  FIGURE 1: LER vs Code Size (THE HEADLINE)
  ==========================================

  Panel (a):                         Panel (b):
  Log-scale LER for each decoder     % improvement over BP
  at each code size.                  for GNN decoders.

    LER                                Improvement
   1e-1 |  x BP                        60% |     ___
        |  x                                |    |   |
   1e-2 |  o GNN-BP                    40% | ___|   |
        |  o       x                        ||      |
   1e-3 |  D       o  x               20% ||       |___
        |  D GNN+  o  o                    |            |
   1e-4 |  LSD     D  D                0% |            |
        +--------+--------+               +--+----+----+
        [[72]]  [[144]]  [[288]]         72  144   288

  This is the single most important figure.
  Shows GNN improvement GROWS with code size.


  FIGURE 2: Decoder Comparison Bars
  =================================

  One panel per code size. All decoders side by side.
  Best decoder highlighted with blue border.

    |        |
    | ____   |  ____
    ||    |  | |    | ____
    ||    |  | |    ||    |
    ||BP  |  | |GNN ||OSD |  ...
    +-----+  +-----------+
    [[72,12,6]]


  FIGURE 3: Threshold Curves (LER vs p)
  ======================================

  Classic QEC plot. One panel per code size.
  Multiple decoders overlaid with error bars.

    LER
   1e-1 |  *---*---*  BP
        |   \
   1e-2 |    *---*---*  GNN-BP
        |     \
   1e-3 |      *---*---*  BP-OSD
        +--+---+---+---+-> p
          0.01  0.03  0.05


  FIGURE 4: Static vs Drift
  ==========================

  Paired bars: solid (static) vs hatched (drift).
  Shows performance degradation under drift.

    LER
        | ____
        ||    |////|
        ||stat|drft|
        |+----+----+
         BP  GNN-BP


  FIGURE 5: Training Dynamics
  ===========================

  4 panels per training run:
    (a) Loss curves (train/val)
    (b) BP convergence rate
    (c) Bit accuracy
    (d) Learning rate schedule


  FIGURE 6: Improvement Heatmap
  =============================

  Matrix: rows = decoders, columns = conditions (code+p+noise)
  Colors: blue = improvement, red = degradation
  Cell values: "+23%", "-5%", etc.

              n=72     n=144    n=288
              p=0.02   p=0.04   p=0.04
    GNN-BP    +15%     +22%     +35%
    Int.GNN   +23%     +31%     +48%
    BP-OSD    +80%     +85%     +90%
    GNN+LSD   +90%     +93%     +96%


  FIGURE 7: Multi-Code Overlay (NEW -- the money plot)
  =====================================================

  One panel PER DECODER.
  All code sizes overlaid on the same axes.
  If lines CROSS, that's the threshold!

    LER (BP panel)           LER (GNN-BP panel)
   1e-1 | o-o-o [[72]]      1e-1 | o-o-o [[72]]
        | s-s-s [[144]]          | s-s-s [[144]]
   1e-2 | D-D-D [[288]]     1e-2 | D-D-D [[288]]
        |   \ \   \               |   \ \ \
   1e-3 |    \ \   \        1e-3 |    X  <- threshold crossing!
        +----+--+--+-->p         +---+--+--+-->p

  Where lines cross = threshold error rate.
  Below threshold: larger code = exponentially better.
```


---

## 15. Circuit-Level Decoding (DEM)

Code-capacity is a simplification. Circuit-level adds measurement errors.

```
  CODE-CAPACITY vs CIRCUIT-LEVEL
  ==============================

  Code-capacity (what we mostly train on):
    - Perfect syndrome measurements
    - Errors only on data qubits
    - Simpler: decode from (n, ) error vector

  Circuit-level (realistic):
    - NOISY syndrome measurements (can have errors too!)
    - Multiple measurement rounds (e.g., 5 rounds)
    - Errors on data qubits AND measurement circuits
    - Much harder: many more error mechanisms


  DETECTOR ERROR MODEL (DEM):
  ===========================

  Stim extracts a DEM from the circuit:

    +----------------+-----+-----+-----+-----+
    |                | e_1 | e_2 | e_3 | ... |  ~3101 error mechanisms
    +----------------+-----+-----+-----+-----+
    | detector_1     |  1  |  0  |  1  | ... |
    | detector_2     |  0  |  1  |  0  | ... |
    | ...            |     |     |     |     |  288 detectors
    | detector_288   |  1  |  0  |  0  | ... |  (for 5 rounds)
    +----------------+-----+-----+-----+-----+

    dem_pcm: (288 detectors x ~3101 errors)  -- sparse!
    error_probs: (~3101,) prior probability for each error mechanism
    obs_matrix: (~3101 x 12) which errors flip which observables

  For [[72,12,6]] with 5 rounds:
    288 detectors, ~3101 error mechanisms, 12 observables


  DEM TANNER GRAPH:
  ================

    Much simpler than CSS Tanner graph:
    - Node type 0: Error variables (~3101 nodes)
    - Node type 1: Detectors (288 nodes)
    - Edge type: only 1 (all edges same type)
    - Features: [prior_llr, 1, 0, 0] for errors,
                [syndrome,  0, 1, 0] for detectors


  CHALLENGE:
  ==========

    BP convergence on DEM is ~0% (coin-flipping!)
    That's why we need GNN + post-processing for circuit-level.
    Observable loss lets us train WITHOUT ground-truth fault patterns.
```


---

## 16. Quick Reference: Pipeline Commands

```bash
# =============================================
#  DATA GENERATION
# =============================================

# Code-capacity data (static noise)
python -m gnn_pipeline.generate_codecap \
    --code 72_12_6 --p 0.02 --eta 20 --shots 5000 \
    --out data/train_72.npz

# Code-capacity data (with drift)
python -m gnn_pipeline.generate_codecap \
    --code 72_12_6 --p 0.02 --eta 20 --shots 5000 \
    --drift_model sine --drift_amp 0.015 \
    --out data/train_72_drift.npz

# Circuit-level data
python -m astra_stim --d 6 --rounds 5 \
    --noise biased_circuit --p 0.005 --eta 20 \
    --shots 1000 --out data/circuit_train.npz


# =============================================
#  TRAINING
# =============================================

# Self-supervised (no ground truth needed)
python -m gnn_pipeline.train_selfsupervised \
    --in_glob "data/*.npz" \
    --hidden_dim 128 --num_mp_layers 5 \
    --use_residual --use_layer_norm --use_attention \
    --epochs 15 --scheduler cosine \
    --out_dir runs/selfsup

# Supervised (with ground-truth error patterns)
python -m gnn_pipeline.train_supervised \
    --in_glob "data/*.npz" \
    --pretrained model.pt \
    --loss focal --curriculum --augment \
    --learnable_alpha --hidden_dim 128 \
    --num_mp_layers 5 --use_residual --use_layer_norm \
    --use_attention --epochs 20 \
    --out_dir runs/supervised

# Unified trainer (supports both modes + FiLM + online regen)
python -m gnn_pipeline.train_unified \
    --mode code_capacity \
    --in_glob "data/*.npz" \
    --loss focal --use_film \
    --online_regen --p_range 0.01 0.04 \
    --tensorboard --epochs 20 \
    --out_dir runs/unified

# Circuit-level training
python -m gnn_pipeline.train_circuit \
    --in_glob "data/circuit_*.npz" \
    --epochs 20 --batch_size 8 --lr 1e-4 \
    --bp_iters 10 --hidden_dim 64 \
    --num_mp_layers 3 \
    --out_dir runs/circuit


# =============================================
#  EVALUATION
# =============================================

# Evaluate all decoders
python -m gnn_pipeline.evaluate \
    --test_npz data/test.npz \
    --bposd --bplsd --belieffind --mwpm \
    --out_dir runs/eval

# With GNN model
python -m gnn_pipeline.evaluate \
    --test_npz data/test.npz \
    --gnn_model runs/supervised/best_model.pt \
    --bposd --bplsd \
    --out_dir runs/eval_gnn


# =============================================
#  THRESHOLD SWEEP
# =============================================

python -m gnn_pipeline.threshold_sweep \
    --p_min 0.01 --p_max 0.06 --num_points 10 \
    --shots 5000 --eta 20 \
    --drift_models sine,ou,rtn --drift_amp 0.02 \
    --bposd --bplsd \
    --out_dir runs/sweep


# =============================================
#  ABLATION STUDY
# =============================================

python -m gnn_pipeline.ablation \
    --train_data "data/train.npz" \
    --test_data "data/test.npz" \
    --epochs 10 \
    --out_dir runs/ablation


# =============================================
#  PUBLICATION FIGURES
# =============================================

# Generate all 7 figures as PDF
python -m gnn_pipeline.make_figures \
    --results_dir runs --out_dir figures --format pdf

# Generate specific figures as PNG
python -m gnn_pipeline.make_figures \
    --results_dir runs --out_dir figures --format png --fig 1 3 7
```

---

## Appendix: Architecture Decision Summary

```
  WHY THESE CHOICES?
  ==================

  +-------------------------+---------------------------------------------+
  | Decision                | Rationale                                   |
  +-------------------------+---------------------------------------------+
  | Min-sum (not sum-prod)  | Numerically stable, hardware-friendly,      |
  |                         | nearly same performance with alpha scaling   |
  +-------------------------+---------------------------------------------+
  | Separate X/Z decoding   | CSS structure requires it; combined matrix   |
  |                         | destroys the code structure                  |
  +-------------------------+---------------------------------------------+
  | GNN before BP           | GNN is fast (~1ms), BP is iterative (~10ms) |
  | (not replacing BP)      | GNN learns WHAT to fix, BP does the work    |
  +-------------------------+---------------------------------------------+
  | FiLM over separate      | One model for all p-values saves memory     |
  | per-p models            | and enables interpolation to unseen p       |
  +-------------------------+---------------------------------------------+
  | Interleaved over        | Intermediate BP state gives GNN more info   |
  | one-shot correction     | than raw channel LLR alone                  |
  +-------------------------+---------------------------------------------+
  | Focal loss over BCE     | Rare errors = imbalanced problem            |
  |                         | Focal auto-focuses on hard examples         |
  +-------------------------+---------------------------------------------+
  | Online regeneration     | Prevents overfitting to fixed dataset       |
  | over fixed data         | Infinite diversity, matches Astra paradigm  |
  +-------------------------+---------------------------------------------+
  | TopK-2 min-sum          | O(k) instead of O(k log k) for CTV         |
  |                         | (only need 2 smallest values)               |
  +-------------------------+---------------------------------------------+
  | LayerNorm on input      | LLR features (~20) vs indicators (~1)       |
  |                         | mixed scale without normalization hurts      |
  +-------------------------+---------------------------------------------+
```
