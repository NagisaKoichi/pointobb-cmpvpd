# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class JSLoss(nn.Module):
    """JS Divergence Loss for Variational Inference.

    This loss measures the Jensen-Shannon divergence between the predicted
    distribution and the ground truth distribution, as used in VPD framework.

    Args:
        reduction (str): The method to reduce the loss. Options are "none",
            "mean" and "sum". Default: "mean".
        loss_weight (float): The weight of the loss. Default: 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(JSLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred_dist,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred_dist (Tensor): Predicted distribution parameters with shape (N, 8).
                First 4 channels are mean, last 4 channels are log std.
            target (Tensor): Ground truth bbox targets with shape (N, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (int, optional): Average factor for loss normalization.
            reduction_override (str, optional): Override reduction method.

        Returns:
            Tensor: Calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # Split predicted distribution into mean and log std
        pred_mean = pred_dist[:, :4]
        pred_lstd = pred_dist[:, 4:]

        # Compute JS divergence
        # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5(P+Q)
        # For Gaussian distributions with diagonal covariance:
        # KL(N(μ1,σ1²)||N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 0.5

        # Assume target has zero variance (delta distribution)
        # Simplify: JS ≈ 0.5 * [log(σ) + (μ-target)²/σ² + σ²]

        pred_std = pred_lstd.exp()
        diff = pred_mean - target

        # JS divergence approximation
        js_div = 0.5 * (
            pred_lstd +  # log(σ)
            (diff.pow(2) / (pred_std.pow(2) + 1e-6)) +  # (μ-target)²/σ²
            pred_std.pow(2) -  # σ²
            1.0  # constant term
        )

        loss = js_div.sum(dim=-1)

        # Apply weights
        if weight is not None:
            loss = loss * weight

        # Apply reduction
        if reduction == 'mean':
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss
