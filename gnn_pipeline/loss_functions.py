"""Loss functions for supervised GNN-BP training.

Provides specialized loss functions that handle the extreme class imbalance
in quantum error correction (errors are rare events at ~1-5% per qubit).

Loss functions:
    - weighted_bce: binary cross-entropy with pos_weight upweighting errors
    - focal: focal loss that down-weights easy negatives
    - logical_aware: wraps any base loss and adds a soft logical-flip penalty
    - coset: degeneracy-aware loss using soft syndrome + logical observable check
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_loss(
    marginals: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 50.0,
) -> torch.Tensor:
    """Weighted binary cross-entropy via logits.

    Converts marginals (probabilities) back to logits and applies
    F.binary_cross_entropy_with_logits with pos_weight to upweight
    rare error positions.

    Args:
        marginals: (B, n) predicted error probabilities from BP
        targets: (B, n) ground-truth binary error vectors
        pos_weight: weight for positive (error) class; ~1/error_rate
    """
    eps = 1e-7
    marginals_c = marginals.clamp(eps, 1.0 - eps)
    logits = torch.logit(marginals_c)  # numerically stable logit
    pw = torch.tensor([pos_weight], device=marginals.device, dtype=marginals.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def focal_loss(
    marginals: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy negatives and focuses on hard examples (misclassified
    error positions).

    Args:
        marginals: (B, n) predicted error probabilities from BP
        targets: (B, n) ground-truth binary error vectors
        alpha: weighting factor for positive class
        gamma: focusing parameter (higher = more focus on hard examples)
    """
    eps = 1e-7
    p = marginals.clamp(eps, 1.0 - eps)

    # p_t = p for positive class, (1-p) for negative class
    p_t = targets * p + (1.0 - targets) * (1.0 - p)

    # alpha_t
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)

    # focal weight
    focal_weight = (1.0 - p_t) ** gamma

    # cross-entropy
    ce = -torch.log(p_t)

    loss = alpha_t * focal_weight * ce
    return loss.mean()


def logical_aware_loss(
    marginals_z: torch.Tensor,
    marginals_x: torch.Tensor,
    targets_z: torch.Tensor,
    targets_x: torch.Tensor,
    lx: torch.Tensor,
    lz: torch.Tensor,
    base_loss_fn,
    logical_weight: float = 0.1,
) -> torch.Tensor:
    """Loss with soft logical-flip penalty.

    Adds a term penalizing predicted logical flip probability deviating
    from the ground-truth logical flip, using the XOR probability formula:
        P(odd parity) = 0.5 * (1 - prod(1 - 2*p_i))

    Args:
        marginals_z: (B, n) predicted Z-error probabilities
        marginals_x: (B, n) predicted X-error probabilities
        targets_z: (B, n) ground-truth Z-error vectors
        targets_x: (B, n) ground-truth X-error vectors
        lx: (k, n) X logical operators (detect Z errors)
        lz: (k, n) Z logical operators (detect X errors)
        base_loss_fn: callable(marginals, targets) -> scalar
        logical_weight: weight for the logical penalty term
    """
    # Base reconstruction loss
    base = base_loss_fn(marginals_z, targets_z) + base_loss_fn(marginals_x, targets_x)

    # Soft logical flip probabilities using XOR formula
    # For Z errors: lx @ z_error gives logical observable
    # P(logical flip from Z) = 0.5 * (1 - prod_j(1 - 2*p_j) for j in support(lx_row))
    def soft_logical_flip(marginals, logicals):
        # marginals: (B, n), logicals: (k, n)
        # Compute in log-domain for numerical stability:
        #   prod(1-2p_j) = exp(sum(log|1-2p_j|)) * sign_product
        # P(flip) = 0.5 * (1 - prod(1-2p_j))
        eps = 1e-7
        p = marginals.clamp(eps, 1.0 - eps)
        factor = 1.0 - 2.0 * p  # (B, n), range (-1, 1)
        log_mask = logicals.unsqueeze(0).float()  # (1, k, n)

        # Log-domain product: sum log|factor| over support
        log_abs_factor = torch.log(factor.abs().clamp(min=eps)).unsqueeze(1)  # (B, 1, n)
        log_prod = (log_abs_factor * log_mask).sum(dim=2)  # (B, k)

        # Sign product over support
        sign_factor = torch.sign(factor).unsqueeze(1)  # (B, 1, n)
        sign_masked = sign_factor * log_mask + (1.0 - log_mask)  # 1.0 for non-support
        sign_prod = sign_masked.prod(dim=2)  # (B, k)

        prod_val = sign_prod * torch.exp(log_prod)
        flip_prob = 0.5 * (1.0 - prod_val)  # (B, k)
        return flip_prob.clamp(0.0, 1.0)

    # Ground-truth logical flips
    gt_z_flip = (targets_z.unsqueeze(1) * lx.unsqueeze(0).float()).sum(dim=2) % 2  # (B, k)
    gt_x_flip = (targets_x.unsqueeze(1) * lz.unsqueeze(0).float()).sum(dim=2) % 2  # (B, k)

    # Predicted logical flip probabilities
    pred_z_flip = soft_logical_flip(marginals_z, lx)
    pred_x_flip = soft_logical_flip(marginals_x, lz)

    logical_loss = F.mse_loss(pred_z_flip, gt_z_flip) + F.mse_loss(pred_x_flip, gt_x_flip)

    return base + logical_weight * logical_loss


def syndrome_consistency_loss(
    marginals: torch.Tensor,
    syndrome: torch.Tensor,
    pcm: torch.Tensor,
) -> torch.Tensor:
    """Soft syndrome consistency: penalize mismatch between predicted errors and syndrome.

    Uses the differentiable XOR probability formula to compute the expected syndrome
    from soft marginals, then penalizes deviation from the actual syndrome.

    This gives the GNN direct gradient signal about whether its corrections lead to
    syndrome-consistent decoding.

    Args:
        marginals: (B, n) soft error probabilities from BP
        syndrome: (B, m) actual syndrome bits
        pcm: (m, n) parity-check matrix
    """
    eps = 1e-7
    p = marginals.clamp(eps, 1.0 - eps)  # (B, n)
    # For each check row, compute P(odd parity) = 0.5 * (1 - prod(1 - 2*p_j))
    # In log domain for stability
    factor = 1.0 - 2.0 * p  # (B, n)
    log_abs = torch.log(factor.abs().clamp(min=eps))  # (B, n)
    pcm_f = pcm.float()  # (m, n)

    # Sum log|1-2p_j| over support of each check
    log_prod = log_abs @ pcm_f.t()  # (B, m)

    # Sign product: even number of negatives -> positive
    sign_factor = torch.sign(factor)  # (B, n)
    # Count negative signs per check (where pcm has 1s)
    # sign(prod) = prod(sign_j) for j in support
    # Use log-domain trick: sum of log(sign) doesn't work, but we can use:
    # sign_prod = (-1)^(number of negative factors in support)
    neg_count = ((sign_factor < 0).float()) @ pcm_f.t()  # (B, m)
    sign_prod = 1.0 - 2.0 * (neg_count % 2)  # (B, m)

    prod_val = sign_prod * torch.exp(log_prod)
    predicted_syn = 0.5 * (1.0 - prod_val)  # (B, m)
    predicted_syn = predicted_syn.clamp(0.0, 1.0)

    return F.binary_cross_entropy(predicted_syn, syndrome)


def coset_loss(
    marginals: torch.Tensor,
    targets: torch.Tensor,
    syndrome: torch.Tensor,
    pcm: torch.Tensor,
    logicals: torch.Tensor,
    syn_weight: float = 1.0,
    logical_weight: float = 1.0,
) -> torch.Tensor:
    """Degeneracy-aware loss for CSS codes.

    Instead of penalizing per-qubit error mismatch (which punishes degenerate
    solutions), this loss only penalizes:
      1. Syndrome mismatch: predicted errors should produce the correct syndrome
      2. Logical observable mismatch: predicted errors should have correct coset

    Uses |sin(pi*x/2)| as a differentiable modulo-2 operation, following
    Astra (arXiv:2408.07038) and quaternary NBP (arXiv:2308.08208).

    Args:
        marginals: (B, n) soft error probabilities from BP
        targets: (B, n) ground-truth binary error vectors
        syndrome: (B, m) actual syndrome bits
        pcm: (m, n) parity-check matrix
        logicals: (k, n) logical operator matrix
        syn_weight: weight for syndrome consistency term
        logical_weight: weight for logical observable term
    """
    eps = 1e-7
    p = marginals.clamp(eps, 1.0 - eps)

    # --- Component 1: Soft syndrome consistency ---
    # Predicted syndrome from marginals using differentiable XOR formula
    # P(odd parity over check row) = 0.5 * (1 - prod(1 - 2*p_j))
    factor = 1.0 - 2.0 * p  # (B, n)
    log_abs = torch.log(factor.abs().clamp(min=eps))
    pcm_f = pcm.float()

    log_prod = log_abs @ pcm_f.t()  # (B, m)
    neg_count = ((factor < 0).float()) @ pcm_f.t()
    sign_prod = 1.0 - 2.0 * (neg_count % 2)

    predicted_syn = 0.5 * (1.0 - sign_prod * torch.exp(log_prod))
    predicted_syn = predicted_syn.clamp(0.0, 1.0)

    syn_loss = F.binary_cross_entropy(predicted_syn, syndrome)

    # --- Component 2: Logical observable consistency ---
    # Ground-truth logical flips: (targets @ logicals.T) mod 2
    logicals_f = logicals.float()
    gt_logical = (targets @ logicals_f.t()) % 2  # (B, k)

    # Predicted logical flip probability using XOR formula
    log_prod_l = log_abs @ logicals_f.t()  # (B, k)
    neg_count_l = ((factor < 0).float()) @ logicals_f.t()
    sign_prod_l = 1.0 - 2.0 * (neg_count_l % 2)

    predicted_logical = 0.5 * (1.0 - sign_prod_l * torch.exp(log_prod_l))
    predicted_logical = predicted_logical.clamp(0.0, 1.0)

    logical_loss = F.binary_cross_entropy(predicted_logical, gt_logical)

    return syn_weight * syn_loss + logical_weight * logical_loss


def constraint_loss(
    marginals: torch.Tensor,
    syndrome: torch.Tensor,
    pcm: torch.Tensor,
    logicals: torch.Tensor,
    targets: torch.Tensor,
    syn_weight: float = 1.0,
    logical_weight: float = 1.0,
) -> torch.Tensor:
    """Differentiable mod-2 constraint loss using |sin(pi*x/2)| surrogate.

    Following Astra (arXiv:2408.07038), uses |sin(pi*x/2)| as a smooth,
    differentiable approximation to (x mod 2). Unlike XOR probability formulas,
    this operates directly on soft error counts and provides strong gradients
    near constraint satisfaction.

    Key property: |sin(pi*x/2)| = 0 when x is even, 1 when x is odd,
    and smoothly interpolates, giving non-zero gradients everywhere.

    Args:
        marginals: (B, n) soft error probabilities from BP
        syndrome: (B, m) actual syndrome bits (0 or 1)
        pcm: (m, n) parity-check matrix
        logicals: (k, n) logical operator matrix
        targets: (B, n) ground-truth binary error vectors
        syn_weight: weight for syndrome constraint term
        logical_weight: weight for logical observable constraint term
    """
    eps = 1e-7
    p = marginals.clamp(eps, 1.0 - eps)  # (B, n)
    pcm_f = pcm.float()  # (m, n)
    logicals_f = logicals.float()  # (k, n)

    # Predicted soft syndrome: sum of marginals per check row
    soft_syn = p @ pcm_f.t()  # (B, m)
    # |sin(pi*x/2)| gives 0 for even parity, 1 for odd parity
    pred_syn = torch.abs(torch.sin(soft_syn * (torch.pi / 2.0)))  # (B, m)
    syn_loss = F.binary_cross_entropy(pred_syn.clamp(eps, 1.0 - eps), syndrome)

    # Ground-truth logical flips: (targets @ logicals.T) mod 2
    gt_logical = torch.fmod(torch.round(targets @ logicals_f.t()), 2.0)  # (B, k)

    # Predicted soft logical flips
    soft_log = p @ logicals_f.t()  # (B, k)
    pred_log = torch.abs(torch.sin(soft_log * (torch.pi / 2.0)))  # (B, k)
    log_loss = F.binary_cross_entropy(pred_log.clamp(eps, 1.0 - eps), gt_logical)

    return syn_weight * syn_loss + logical_weight * log_loss


def per_iteration_loss(
    marginals_list: list,
    targets: torch.Tensor,
    base_loss_fn,
    decay: float = 0.8,
) -> torch.Tensor:
    """Compute weighted sum of losses at each BP iteration.

    Encourages the GNN to produce corrections that improve decoding
    at every stage, not just the final iteration. Later iterations
    get higher weight (geometric decay from first to last).

    Args:
        marginals_list: list of (B, n) marginals from each BP iteration
        targets: (B, n) ground-truth binary error vectors
        base_loss_fn: callable(marginals, targets) -> scalar loss
        decay: geometric weight factor; final iteration gets weight 1.0,
               second-to-last gets decay, etc.
    """
    T = len(marginals_list)
    if T == 0:
        return torch.tensor(0.0, device=targets.device, dtype=targets.dtype,
                            requires_grad=True)

    total = torch.tensor(0.0, device=marginals_list[0].device)
    weight_sum = 0.0
    for t, marg in enumerate(marginals_list):
        w = decay ** (T - 1 - t)  # last iteration gets weight 1.0
        total = total + w * base_loss_fn(marg, targets)
        weight_sum += w

    return total / weight_sum


def observable_loss(
    marginals: torch.Tensor,
    obs_matrix: torch.Tensor,
    true_obs: torch.Tensor,
    syndrome: torch.Tensor,
    dem_pcm: torch.Tensor,
    syn_weight: float = 1.0,
    obs_weight: float = 1.0,
) -> torch.Tensor:
    """Circuit-level loss: penalizes wrong observables and syndrome mismatch.

    For circuit-level decoding, we don't have ground-truth fault patterns.
    Instead we use:
      1. Observable loss: predicted observable flips should match actual
      2. Syndrome loss: predicted faults should explain detector events

    Uses the XOR probability formula for differentiable mod-2:
        P(odd parity) = 0.5 * (1 - prod(1 - 2*p_i))

    Args:
        marginals: (B, num_errors) soft fault probabilities from BP
        obs_matrix: (num_errors, num_observables) which observables each fault flips
        true_obs: (B, num_observables) actual observable outcomes
        syndrome: (B, num_detectors) detector events
        dem_pcm: (num_detectors, num_errors) DEM parity-check matrix
        syn_weight: weight for syndrome consistency term
        obs_weight: weight for observable prediction term
    """
    eps = 1e-7
    p = marginals.clamp(eps, 1.0 - eps)  # (B, E)
    factor = 1.0 - 2.0 * p  # (B, E)
    log_abs = torch.log(factor.abs().clamp(min=eps))  # (B, E)

    # --- Observable loss ---
    # For each observable k, compute P(flip) over faults in obs_matrix[:, k]
    obs_f = obs_matrix.float()  # (E, num_obs)
    log_prod_obs = log_abs @ obs_f  # (B, num_obs)
    neg_count_obs = ((factor < 0).float()) @ obs_f
    sign_prod_obs = 1.0 - 2.0 * (neg_count_obs % 2)

    pred_obs = 0.5 * (1.0 - sign_prod_obs * torch.exp(log_prod_obs))
    pred_obs = pred_obs.clamp(0.0, 1.0)  # (B, num_obs)

    obs_loss = F.binary_cross_entropy(pred_obs, true_obs)

    # --- Syndrome consistency loss ---
    # Predicted detector events from marginals
    pcm_f = dem_pcm.float()  # (D, E)
    log_prod_syn = log_abs @ pcm_f.t()  # (B, D)
    neg_count_syn = ((factor < 0).float()) @ pcm_f.t()
    sign_prod_syn = 1.0 - 2.0 * (neg_count_syn % 2)

    pred_syn = 0.5 * (1.0 - sign_prod_syn * torch.exp(log_prod_syn))
    pred_syn = pred_syn.clamp(0.0, 1.0)  # (B, D)

    syn_loss = F.binary_cross_entropy(pred_syn, syndrome)

    return obs_weight * obs_loss + syn_weight * syn_loss


def observable_loss_decomposed(
    marginals: torch.Tensor,
    obs_matrix: torch.Tensor,
    true_obs: torch.Tensor,
    syndrome: torch.Tensor,
    dem_pcm: torch.Tensor,
    syn_weight: float = 1.0,
    obs_weight: float = 1.0,
    obs_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Circuit-level loss with per-observable decomposition.

    Like ``observable_loss`` but returns both the total loss and the
    per-observable loss vector so that the training loop can:
      - Log per-observable losses for diagnostics
      - Dynamically weight harder observables higher

    Args:
        marginals: (B, num_errors) soft fault probabilities from BP
        obs_matrix: (num_errors, num_observables) which observables each fault flips
        true_obs: (B, num_observables) actual observable outcomes
        syndrome: (B, num_detectors) detector events
        dem_pcm: (num_detectors, num_errors) DEM parity-check matrix
        syn_weight: weight for syndrome consistency term
        obs_weight: weight for observable prediction term
        obs_weights: optional (num_observables,) per-observable weights.
            If *None*, all observables are weighted equally.

    Returns:
        (total_loss, per_obs_losses):
            total_loss: scalar combined loss
            per_obs_losses: (num_observables,) detached per-observable BCE losses
    """
    eps = 1e-7
    p = marginals.clamp(eps, 1.0 - eps)
    factor = 1.0 - 2.0 * p
    log_abs = torch.log(factor.abs().clamp(min=eps))

    # --- Per-observable loss ---
    obs_f = obs_matrix.float()
    log_prod_obs = log_abs @ obs_f
    neg_count_obs = ((factor < 0).float()) @ obs_f
    sign_prod_obs = 1.0 - 2.0 * (neg_count_obs % 2)

    pred_obs = 0.5 * (1.0 - sign_prod_obs * torch.exp(log_prod_obs))
    pred_obs = pred_obs.clamp(eps, 1.0 - eps)  # (B, num_obs)

    num_obs = true_obs.shape[1]
    per_obs = torch.stack([
        F.binary_cross_entropy(pred_obs[:, k], true_obs[:, k])
        for k in range(num_obs)
    ])  # (num_obs,)

    if obs_weights is not None:
        weighted_obs_loss = (per_obs * obs_weights).sum() / obs_weights.sum().clamp(min=eps)
    else:
        weighted_obs_loss = per_obs.mean()

    # --- Syndrome consistency loss ---
    pcm_f = dem_pcm.float()
    log_prod_syn = log_abs @ pcm_f.t()
    neg_count_syn = ((factor < 0).float()) @ pcm_f.t()
    sign_prod_syn = 1.0 - 2.0 * (neg_count_syn % 2)

    pred_syn = 0.5 * (1.0 - sign_prod_syn * torch.exp(log_prod_syn))
    pred_syn = pred_syn.clamp(eps, 1.0 - eps)

    syn_loss = F.binary_cross_entropy(pred_syn, syndrome)

    total = obs_weight * weighted_obs_loss + syn_weight * syn_loss
    return total, per_obs.detach()
