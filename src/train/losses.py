"""Defines L_f (fusion BCE), unimodal BCE terms, L_rel (KL to detached soft target), optional L_con (InfoNCE), and L_bal — Section 8.

Status: implemented — provides individual loss functions and CombinedLoss that
computes the full objective L = L_f + α(BCE_v + BCE_a) + β·L_rel + γ·L_con + η·L_bal
with configurable weights. See AGENTS.md for build order.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def fusion_bce_loss(
    logit_f: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Fusion binary cross-entropy loss L_f = BCE(y, p_f).

    Parameters
    ----------
    logit_f:
        Fused logit ``(B, 1)``.
    target:
        Binary target ``(B, 1)`` or ``(B,)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    target = target.float().view_as(logit_f)
    return F.binary_cross_entropy_with_logits(logit_f, target)


def unimodal_bce_loss(
    logit_v: torch.Tensor,
    logit_a: torch.Tensor,
    target: torch.Tensor,
    visual_missing: torch.Tensor | None = None,
    audio_missing: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unimodal BCE losses for visual and audio heads.

    Parameters
    ----------
    logit_v:
        Visual logit ``(B, 1)``.
    logit_a:
        Audio logit ``(B, 1)``.
    target:
        Binary target ``(B, 1)`` or ``(B,)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(loss_v, loss_a, combined)`` where combined = loss_v + loss_a.
    """
    target = target.float().view(-1, 1)
    per_v = F.binary_cross_entropy_with_logits(logit_v, target, reduction="none")
    per_a = F.binary_cross_entropy_with_logits(logit_a, target, reduction="none")
    mask_v = (
        torch.ones_like(per_v, dtype=torch.bool)
        if visual_missing is None
        else visual_missing.reshape(-1, 1).to(per_v.device) < 0.5
    )
    mask_a = (
        torch.ones_like(per_a, dtype=torch.bool)
        if audio_missing is None
        else audio_missing.reshape(-1, 1).to(per_a.device) < 0.5
    )
    loss_v = (per_v * mask_v.to(per_v.dtype)).sum() / mask_v.sum().clamp_min(1)
    loss_a = (per_a * mask_a.to(per_a.dtype)).sum() / mask_a.sum().clamp_min(1)
    return loss_v, loss_a, loss_v + loss_a


def reliability_loss(
    weights: torch.Tensor,
    logit_v: torch.Tensor,
    logit_a: torch.Tensor,
    target: torch.Tensor,
    tau_r: float = 1.0,
    visual_missing: torch.Tensor | None = None,
    audio_missing: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reliability loss L_rel = KL(t || w) per Section 7.2.

    The detached soft target is:
        t_m = exp(-stopgrad(ℓ_m) / τ_r) / Σ_j exp(-stopgrad(ℓ_j) / τ_r)

    where ℓ_m is the per-sample unimodal BCE loss for modality m.

    Parameters
    ----------
    weights:
        Gate weights ``(B, 2)`` = ``[w_v, w_a]``.
    logit_v:
        Visual logit ``(B, 1)``.
    logit_a:
        Audio logit ``(B, 1)``.
    target:
        Binary target ``(B, 1)`` or ``(B,)``.
    tau_r:
        Temperature for the soft target (default 1.0).

    Returns
    -------
    torch.Tensor
        Scalar KL divergence loss.
    """
    target = target.float().view(-1, 1)

    # Per-sample unimodal losses (no reduction)
    loss_v = F.binary_cross_entropy_with_logits(
        logit_v, target, reduction="none"
    ).squeeze(-1)  # (B,)
    loss_a = F.binary_cross_entropy_with_logits(
        logit_a, target, reduction="none"
    ).squeeze(-1)  # (B,)

    missing_v = (
        torch.zeros_like(loss_v, dtype=torch.bool)
        if visual_missing is None
        else visual_missing.reshape(-1).to(loss_v.device) >= 0.5
    )
    missing_a = (
        torch.zeros_like(loss_a, dtype=torch.bool)
        if audio_missing is None
        else audio_missing.reshape(-1).to(loss_a.device) >= 0.5
    )
    both_missing = missing_v & missing_a

    # Detached soft targets over available modalities only.
    neg_losses = torch.stack([-loss_v, -loss_a], dim=-1).detach()  # (B, 2)
    neg_losses[:, 0] = neg_losses[:, 0].masked_fill(missing_v, -torch.inf)
    neg_losses[:, 1] = neg_losses[:, 1].masked_fill(missing_a, -torch.inf)
    neg_losses[both_missing] = 0.0
    soft_targets = F.softmax(neg_losses / tau_r, dim=-1)  # (B, 2)

    # KL(t || w) = Σ t_m * log(t_m / w_m)
    # Use F.kl_div which expects log(w) as input
    log_weights = torch.log(weights + 1e-8)  # (B, 2)
    per_sample_kl = F.kl_div(log_weights, soft_targets, reduction="none").sum(dim=-1)
    valid = (~both_missing).to(per_sample_kl.dtype)
    kl = (per_sample_kl * valid).sum() / valid.sum().clamp_min(1.0)

    return kl


def contrastive_loss(
    proj_v: torch.Tensor,
    proj_a: torch.Tensor,
    is_real: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Optional InfoNCE contrastive loss L_con on real, aligned pairs.

    Matching audio/video temporal bins from real clips are positives;
    other clips/bins are negatives.  Omitted if it harms the small-data setting.

    Parameters
    ----------
    proj_v:
        Projected visual embeddings ``(B, T, D)`` — only real samples used.
    proj_a:
        Projected audio embeddings ``(B, T, D)``.
    is_real:
        Boolean mask ``(B,)`` indicating real samples.
    temperature:
        InfoNCE temperature (default 0.07).

    Returns
    -------
    torch.Tensor
        Scalar contrastive loss (0 if no real samples in batch).
    """
    # Only use real, aligned pairs
    real_mask = is_real.bool()
    if real_mask.sum() < 2:
        return torch.tensor(0.0, device=proj_v.device)

    v = proj_v[real_mask]  # (N_real, T, D)
    a = proj_a[real_mask]

    # Pool over time for simplicity
    v = F.normalize(v.mean(dim=1), dim=-1)  # (N_real, D)
    a = F.normalize(a.mean(dim=1), dim=-1)  # (N_real, D)

    N = v.shape[0]

    # Cosine similarity matrix
    sim_va = v @ a.T / temperature  # (N, N)
    sim_av = a @ v.T / temperature

    # Labels: diagonal is the positive pair
    labels = torch.arange(N, device=v.device)

    loss_va = F.cross_entropy(sim_va, labels)
    loss_av = F.cross_entropy(sim_av, labels)

    return (loss_va + loss_av) / 2


def balance_loss(
    weights: torch.Tensor,
) -> torch.Tensor:
    """Balance regularization L_bal = Σ_m (w_m - 0.5)^2.

    Used only for brief warm-up or diagnosed collapse.  Must not force equal
    weights after training (Section 8).

    Parameters
    ----------
    weights:
        Gate weights ``(B, 2)`` = ``[w_v, w_a]``.

    Returns
    -------
    torch.Tensor
        Scalar balance loss.
    """
    return ((weights - 0.5) ** 2).sum(dim=-1).mean()


class CombinedLoss(nn.Module):
    """Combined training objective per Section 8.

    L = L_f + α(BCE_v + BCE_a) + β·L_rel + γ·L_con + η·L_bal

    Parameters
    ----------
    alpha:
        Unimodal head weight (default 0.25).
    beta:
        Reliability loss weight (default 0.10).
    gamma:
        Contrastive loss weight (default 0.05).
    eta:
        Balance regularization weight (default 0.0).
    tau_r:
        Temperature for reliability soft target (default 1.0).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 0.10,
        gamma: float = 0.05,
        eta: float = 0.0,
        tau_r: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.tau_r = tau_r

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        target: torch.Tensor,
        proj_v: torch.Tensor | None = None,
        proj_a: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the full combined loss.

        Parameters
        ----------
        model_output:
            Output dict from FullModel.forward().
        target:
            Binary target ``(B,)`` or ``(B, 1)``.
        proj_v:
            Projected visual embeddings for L_con (optional).
        proj_a:
            Projected audio embeddings for L_con (optional).

        Returns
        -------
        dict[str, torch.Tensor]
            - ``"total"``: total combined loss
            - ``"L_f"``: fusion BCE
            - ``"L_unimodal"``: combined unimodal BCE
            - ``"L_rel"``: reliability KL loss
            - ``"L_con"``: contrastive loss
            - ``"L_bal"``: balance regularization
        """
        target = target.float().view(-1, 1)
        visual_missing = model_output.get("visual_missing")
        audio_missing = model_output.get("audio_missing")
        logit_f = model_output["logit_f"]
        has_input = torch.ones_like(target, dtype=torch.bool)
        if visual_missing is not None and audio_missing is not None:
            has_input = ~(
                (visual_missing.reshape(-1, 1).to(target.device) >= 0.5)
                & (audio_missing.reshape(-1, 1).to(target.device) >= 0.5)
            )
        per_fusion = F.binary_cross_entropy_with_logits(logit_f, target, reduction="none")
        l_f = (per_fusion * has_input.to(per_fusion.dtype)).sum() / has_input.sum().clamp_min(1)

        # Unimodal BCE
        _, _, l_unimodal = unimodal_bce_loss(
            model_output["logit_v"], model_output["logit_a"], target,
            visual_missing=visual_missing,
            audio_missing=audio_missing,
        )

        # L_rel: reliability loss
        l_rel = torch.tensor(0.0, device=l_f.device)
        if self.beta > 0 and "w_v" in model_output and "w_a" in model_output:
            weights = torch.cat(
                [model_output["w_v"], model_output["w_a"]], dim=-1
            )
            l_rel = reliability_loss(
                weights, model_output["logit_v"], model_output["logit_a"],
                target, tau_r=self.tau_r,
                visual_missing=visual_missing,
                audio_missing=audio_missing,
            )

        # L_con: contrastive loss
        l_con = torch.tensor(0.0, device=l_f.device)
        if self.gamma > 0 and proj_v is not None and proj_a is not None:
            is_real = (target.view(-1) == 0).float()
            l_con = contrastive_loss(proj_v, proj_a, is_real)

        # L_bal: balance regularization
        l_bal = torch.tensor(0.0, device=l_f.device)
        if self.eta > 0:
            weights = torch.cat(
                [model_output["w_v"], model_output["w_a"]], dim=-1
            )
            l_bal = balance_loss(weights)

        # Combined
        total = (
            l_f
            + self.alpha * l_unimodal
            + self.beta * l_rel
            + self.gamma * l_con
            + self.eta * l_bal
        )

        return {
            "total": total,
            "L_f": l_f,
            "L_unimodal": l_unimodal,
            "L_rel": l_rel,
            "L_con": l_con,
            "L_bal": l_bal,
        }
