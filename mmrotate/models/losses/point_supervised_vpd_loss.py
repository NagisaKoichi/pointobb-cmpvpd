# Copyright (c) OpenMMLab. All rights reserved.
"""Point-Supervised VPD Loss in VPD-style distribution matching.

All computations are done in stride-normalized space for XY offsets only:
    - bbox_mu = (delta_x / stride, delta_y / stride)
    - bbox_log_sigma = log std of (delta_x / stride, delta_y / stride)
    - gt_delta_norm = (gt_center - anchor) / stride

Unlike explicit-prior ELBO variants, this module matches predicted Gaussian
distributions to projected target distributions with Jensen-Shannon divergence,
following VPD-style supervision (without explicit KL-to-prior).
"""
import torch
import torch.nn as nn

from ..builder import ROTATED_LOSSES

NAN_TO_NUM = False


@ROTATED_LOSSES.register_module()
class PointSupervisedVPDLoss(nn.Module):
    """Point-Supervised VPD Loss with VPD-style JS distribution matching.

    Args:
        lambda_center (float): Weight for JS center loss. Default: 1.0.
        project_min (float): Minimum support value for projected distribution.
        project_max (float): Maximum support value for projected distribution.
        num_bins (int): Number of support bins.
        eps (float): Numerical stability epsilon for probabilities.
    """

    def __init__(self,
                 lambda_center=1.0,
                 project_min=-16.0,
                 project_max=16.0,
                 num_bins=33,
                 eps=1e-6,
                 **kwargs):
        super(PointSupervisedVPDLoss, self).__init__()
        self.lambda_center = lambda_center
        self.project_min = float(project_min)
        self.project_max = float(project_max)
        self.num_bins = int(num_bins)
        self.eps = float(eps)
        self.log_sigma_min = -7.0
        self.log_sigma_max = 5.0
        self.sigma_max = 1e2
        if self.num_bins < 2:
            raise ValueError('num_bins must be >= 2 for projected distribution.')

        support = torch.linspace(
            self.project_min, self.project_max, steps=self.num_bins,
            dtype=torch.float32)
        self.register_buffer('support', support)
        self.bin_width = (self.project_max - self.project_min) / (self.num_bins - 1)

    def _sanitize_log_sigma(self, bbox_log_sigma):
        """
        clamp log_sigma to [log_sigma_min, log_sigma_max] and replace NaN/inf with finite values.
        Args:
            bbox_log_sigma (Tensor): (N, 2) predicted log-sigma for XY.
        Returns:
            Tensor: sanitized log-sigma.
        """
        
        bbox_log_sigma = bbox_log_sigma.clamp(
            min=self.log_sigma_min, max=self.log_sigma_max)
        bbox_log_sigma = torch.nan_to_num(
            bbox_log_sigma,
            nan=0.0,
            posinf=self.log_sigma_max,
            neginf=self.log_sigma_min)
        return bbox_log_sigma

    def _project_target_dist(self, target):
        """Project scalar target values onto linear-interpolated support bins.

        Args:
            target (Tensor): (N,) scalar normalized target for one dimension.

        Returns:
            Tensor: (N, K) projected target distribution.
        """
        target = target.clamp(min=self.project_min, max=self.project_max)
        pos = (target - self.project_min) / self.bin_width
        left = torch.floor(pos).long().clamp(min=0, max=self.num_bins - 1)
        right = (left + 1).clamp(max=self.num_bins - 1)

        w_right = (pos - left.float()).clamp(min=0.0, max=1.0)
        w_left = 1.0 - w_right
        same = (left == right)
        w_left = torch.where(same, torch.ones_like(w_left), w_left)
        w_right = torch.where(same, torch.zeros_like(w_right), w_right)

        target_dist = target.new_zeros(target.shape[0], self.num_bins)
        target_dist.scatter_add_(1, left.unsqueeze(1), w_left.unsqueeze(1))
        target_dist.scatter_add_(1, right.unsqueeze(1), w_right.unsqueeze(1))
        target_dist = target_dist.clamp(min=self.eps)
        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True)
        return target_dist

    def _gaussian_binned_dist(self, mu, sigma):
        """Discretize Gaussian(mu, sigma^2) into support bins via CDF masses."""
        k = self.num_bins
        support = self.support.to(mu.dtype)
        edges = torch.empty(k + 1, device=mu.device, dtype=mu.dtype)
        edges[0] = support[0] - 0.5 * self.bin_width
        edges[-1] = support[-1] + 0.5 * self.bin_width
        edges[1:-1] = 0.5 * (support[:-1] + support[1:])

        sigma = sigma.clamp(min=1e-6, max=self.sigma_max)
        z = (edges.unsqueeze(0) - mu.unsqueeze(1)) / (sigma.unsqueeze(1) * (2.0 ** 0.5))
        cdf = 0.5 * (1.0 + torch.erf(z))
        pred_dist = (cdf[:, 1:] - cdf[:, :-1]).clamp(min=self.eps)
        pred_dist = pred_dist / pred_dist.sum(dim=1, keepdim=True)
        return pred_dist

    def _js_divergence(self, p, q):
        """Jensen-Shannon divergence for row-wise distributions."""
        m = 0.5 * (p + q)
        kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=1)
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=1)
        return 0.5 * (kl_pm + kl_qm)
    
    def _kl_divergence(self, p, q):
        """KL divergence D_KL(p || q) for row-wise distributions."""
        kl = (p * (torch.log(p) - torch.log(q))).sum(dim=1)
        return kl
    
    def _wasserstein_distance(self, p, q):
        """1D Wasserstein distance for row-wise distributions."""
        cdf_p = torch.cumsum(p, dim=1)
        cdf_q = torch.cumsum(q, dim=1)
        wasserstein = torch.sum(torch.abs(cdf_p - cdf_q) * self.bin_width, dim=1)
        return wasserstein

    def forward(self,
                bbox_mu,
                bbox_log_sigma,
                pos_points,
                pos_strides,
                gt_centers,
                gt_centers_list,
                cur_iter=0,
                pos_img_ids=None,
                num_samples=None,
                prior_log_sigma=None):
        """Compute VPD-style XY distribution matching loss.

        Args:
            bbox_mu (Tensor): Posterior mean (N, 2).
                [:, :2] = (delta_x/stride, delta_y/stride)  -- normalized center offset
            bbox_log_sigma (Tensor): Posterior log-std (N, 2).
            pos_points (Tensor): Anchor points in image coords (N, 2).
            pos_strides (Tensor): Stride per positive sample (N,).
            gt_centers (Tensor): Matched GT center in image coords (N, 2).
            gt_centers_list (list[Tensor]): Unused, kept for API compatibility.
            cur_iter (int): Current training iteration.

        Returns:
            dict[str, Tensor]: loss_center, loss_kl, loss_var, loss_total.
        """
        N = bbox_mu.shape[0]
        if N == 0:
            zero = bbox_mu.sum() * 0.0
            return dict(loss_center=zero, loss_kl=zero, loss_var=zero, loss_total=zero)

        if NAN_TO_NUM:
            bbox_mu = torch.nan_to_num(bbox_mu, nan=0.0, posinf=1e4, neginf=-1e4)
        bbox_log_sigma = self._sanitize_log_sigma(bbox_log_sigma)

        sigma_q = bbox_log_sigma.exp().clamp(min=1e-6, max=self.sigma_max)
        if NAN_TO_NUM:
            sigma_q = torch.nan_to_num(sigma_q, nan=1.0, posinf=self.sigma_max, neginf=1e-6)

        stride_2d = pos_strides.unsqueeze(1)  # (N, 1)

        # target: (gt_center - anchor) / stride
        gt_delta_norm = (gt_centers - pos_points) / stride_2d  # (N, 2)

        # VPD-style: for each dimension (x,y), project scalar target to support bins,
        # then match predicted Gaussian-binned distribution with JS divergence.
        target_x = self._project_target_dist(gt_delta_norm[:, 0])
        target_y = self._project_target_dist(gt_delta_norm[:, 1])
        pred_x = self._gaussian_binned_dist(bbox_mu[:, 0], sigma_q[:, 0])
        pred_y = self._gaussian_binned_dist(bbox_mu[:, 1], sigma_q[:, 1])

        js_x = self._js_divergence(target_x, pred_x)
        js_y = self._js_divergence(target_y, pred_y)
        # js_x = self._kl_divergence(target_x, pred_x)
        # js_y = self._kl_divergence(target_y, pred_y)
        l_center = 0.5 * (js_x + js_y).mean()
                
        if NAN_TO_NUM:
            l_center = torch.nan_to_num(l_center, nan=0.0, posinf=1e4, neginf=0.0)
        
   
        # VPD mode: no explicit prior/KL branch.
        l_kl = bbox_mu.sum() * 0.0
        l_var = bbox_mu.sum() * 0.0

        loss_total = self.lambda_center * l_center
        if NAN_TO_NUM:
            loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=0.0)

        return dict(
            loss_center=l_center,
            loss_kl=l_kl,
            loss_var=l_var,
            loss_total=loss_total,
        )
