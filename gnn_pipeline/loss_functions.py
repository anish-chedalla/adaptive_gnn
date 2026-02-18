"""Loss functions for supervised GNN-BP training.

Provides specialized loss functions that handle the extreme class imbalance
in quantum error correction (errors are rare events at ~1-5% per qubit).

Loss functions:
    - weighted_bce: binary cross-entropy with pos_weight upweighting errors
    - focal: focal loss that down-weights easy negatives
    - logical_aware: wraps any base loss and adds a soft logical-flip penalty
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
