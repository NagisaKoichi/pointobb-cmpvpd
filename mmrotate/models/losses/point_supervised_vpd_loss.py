# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class PointSupervisedVPDLoss(nn.Module):
    """Point-Supervised VPD Loss for variational inference (bbox only, no angle).

    This loss is designed for point-supervised object detection where only
    GT center points are available (no complete bbox annotations).

    It combines three components:
    1. Center regression loss: Predicted bbox center should match GT center
    2. Uncertainty regularization: Encourage low variance (high confidence)
    3. KL divergence regularization: Prevent distribution collapse

    Args:
        center_weight (float): Weight for center regression loss. Default: 1.0.
        uncertainty_weight (float): Weight for uncertainty regularization. Default: 0.1.
        kl_weight (float): Weight for KL divergence regularization. Default: 0.01.
        prior_mean (float): Prior distribution mean. Default: 0.0.
        prior_std (float): Prior distribution std. Default: 1.0.
        reduction (str): Reduction method. Default: 'mean'.
        loss_weight (float): Overall loss weight. Default: 1.0.
    """

    def __init__(self,
                 center_weight=1.0,
                 uncertainty_weight=0.1,
                 kl_weight=0.01,
                 prior_mean=0.0,
                 prior_std=1.0):
        super(PointSupervisedVPDLoss, self).__init__()
        self.center_weight = center_weight
        self.uncertainty_weight = uncertainty_weight
        self.kl_weight = kl_weight
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def center_regression_loss(self, pred_dist, points, gt_centers):
        """Compute center regression loss.

        Args:
            pred_dist (Tensor): Predicted distribution (N, 8).
                First 4 channels: bbox mean [left, top, right, bottom]
                Last 4 channels: bbox log_std
            points (Tensor): Sample points (N, 2).
            gt_centers (Tensor): GT center points (N, 2).

        Returns:
            Tensor: Center regression loss.
        """
        # Decode predicted bbox center from [l, t, r, b] distances to point.
        # Center offset from point is ((r - l) / 2, (b - t) / 2).
        pred_ltrb = pred_dist[:, :4]
        pred_bbox_center = torch.stack([
            points[:, 0] + 0.5 * (pred_ltrb[:, 2] - pred_ltrb[:, 0]),
            points[:, 1] + 0.5 * (pred_ltrb[:, 3] - pred_ltrb[:, 1])
        ], dim=-1)

        # Ensure gt centers use xy only.
        if gt_centers.numel() == 0:
            return pred_bbox_center.new_zeros(pred_bbox_center.size(0))
        if gt_centers.size(-1) != 2:
            gt_centers = gt_centers[:, :2]

        # Match each positive sample to a GT center when counts differ.
        # This keeps point-supervised training stable across images.
        if gt_centers.size(0) != pred_bbox_center.size(0):
            dists = torch.cdist(points, gt_centers, p=2)
            nearest_gt_inds = dists.argmin(dim=1)
            matched_gt_centers = gt_centers[nearest_gt_inds]
        else:
            matched_gt_centers = gt_centers

        # Compute L1 loss between predicted and GT centers
        loss = F.smooth_l1_loss(pred_bbox_center, matched_gt_centers, reduction='none', beta=1.0)
        loss = loss.sum(dim=-1)

        return loss

    def uncertainty_regularization_loss(self, pred_dist):
        """Compute uncertainty regularization loss.

        Encourages the model to produce low variance (high confidence) predictions.

        Args:
            pred_dist (Tensor): Predicted distribution (N, 8).
                First 4 channels: bbox mean
                Last 4 channels: bbox log_std

        Returns:
            Tensor: Uncertainty regularization loss.
        """
        # Extract log_std: bbox only (4 channels)
        pred_lstd = pred_dist[:, 4:]

        # Convert to std and compute mean
        pred_std = pred_lstd.exp()
        loss = pred_std.mean(dim=-1)

        return loss

    def kl_divergence_loss(self, pred_dist):
        """Compute KL divergence regularization loss.

        KL(q||p) where q is predicted distribution, p is prior N(prior_mean, prior_std²).

        Args:
            pred_dist (Tensor): Predicted distribution (N, 8).
                First 4 channels: bbox mean
                Last 4 channels: bbox log_std

        Returns:
            Tensor: KL divergence loss.
        """
        # Extract mean and log_std
        pred_mean = pred_dist[:, :4]  # bbox only (4 channels)
        pred_lstd = pred_dist[:, 4:]
        pred_std = pred_lstd.exp()

        # KL(N(μ, σ²) || N(μ0, σ0²))
        # = log(σ0/σ) + (σ² + (μ-μ0)²)/(2σ0²) - 0.5
        kl = torch.log(self.prior_std / (pred_std + 1e-6)) + \
             (pred_std.pow(2) + (pred_mean - self.prior_mean).pow(2)) / (2 * self.prior_std ** 2) - 0.5

        loss = kl.sum(dim=-1)

        return loss

    def forward(self,
                pred_dist,
                points,
                gt_centers):
        """Forward function.

        Args:
            pred_dist (Tensor): Predicted distribution (N, 8).
                First 4 channels: bbox mean
                Last 4 channels: bbox log_std
            points (Tensor): Sample points (N, 2).
            gt_centers (Tensor): GT center points (N, 2).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (int, optional): Average factor for loss normalization.
            reduction_override (str, optional): Override reduction method.

        Returns:
            dict: Dictionary of calculated losses.
        """

        # Compute individual losses
        loss_center = self.center_regression_loss(pred_dist, points, gt_centers)
        loss_uncertainty = self.uncertainty_regularization_loss(pred_dist)
        loss_kl = self.kl_divergence_loss(pred_dist)

        # Combine losses
        loss_sum = self.center_weight * loss_center + \
                   self.uncertainty_weight * loss_uncertainty + \
                   self.kl_weight * loss_kl
        loss = {
            'center': loss_center,
            'uncertainty': loss_uncertainty,
            'kl_weight': loss_kl,
            'sum': loss_sum
        }

        return loss
