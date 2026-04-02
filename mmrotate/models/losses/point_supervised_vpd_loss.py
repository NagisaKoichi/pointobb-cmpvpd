# Copyright (c) OpenMMLab. All rights reserved.
"""Point-Supervised VPD Loss with Point-Conditioned Prior.

All computations are done in **stride-normalized space**:
  - bbox_mu[:, :2] = (delta_x / stride, delta_y / stride)  [network output]
  - bbox_mu[:, 2:] = (log_w, log_h)  [log of size, no stride factor]
  - gt_center_delta = (gt_center - anchor) / stride  [target in same space]
  - d_i_norm = d_i_pixels / stride  [kNN distance normalized]

This ensures center loss, KL prior, and posterior are all in the same units.

ELBO objective:
  L = lambda_center * L_center
    + lambda_kl(t) * KL(q_phi || p_psi)
    + lambda_var * L_var

Curriculum:
  Stage A (iter < warmup_iters): lambda_kl = lambda_kl_warmup, sigma_s = sigma_s_init
  Stage B (iter >= warmup_iters): lambda_kl linearly increases to lambda_kl,
                                   sigma_s linearly anneals to sigma_s_final
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class PointSupervisedVPDLoss(nn.Module):
    """Point-Supervised VPD Loss with point-conditioned prior in normalized space.

    Args:
        lambda_center (float): Weight for center regression loss. Default: 1.0.
        lambda_kl (float): Final KL weight (stage B). Default: 0.1.
        lambda_kl_warmup (float): Initial KL weight (stage A). Default: 0.02.
        lambda_var (float): Variance regularization weight. Default: 0.01.
        knn_k (int): Nearest neighbors for density estimation. Default: 5.
        sigma_c_coeff (float): Center prior sigma = sigma_c_coeff * d_i_norm.
            Default: 0.5.
        scale_alpha_w (float): Scale prior mu_w = log(alpha_w * d_i_norm). Default: 1.0.
        scale_alpha_h (float): Scale prior mu_h = log(alpha_h * d_i_norm). Default: 1.0.
        sigma_s_init (float): Initial scale prior sigma. Default: 1.0.
        sigma_s_final (float): Final scale prior sigma (after annealing). Default: 0.4.
        warmup_iters (int): Iterations for stage A. Default: 2000.
        anneal_iters (int): Iterations over which to anneal from stage A to B.
            Default: 2000.
        prior_delta_min (float): Min d_i_norm clamp (in normalized units). Default: 0.5.
        prior_delta_max (float): Max d_i_norm clamp. Default: 16.0.
        kl_clip (float): Hard clip on per-sample KL to prevent spikes. Default: 50.0.
    """

    def __init__(self,
                 lambda_center=1.0,
                 lambda_kl=0.1,
                 lambda_kl_warmup=0.02,
                 lambda_var=0.01,
                 knn_k=5,
                 sigma_c_coeff=0.5,
                 scale_alpha_w=1.0,
                 scale_alpha_h=1.0,
                 sigma_s_init=1.0,
                 sigma_s_final=0.4,
                 warmup_iters=2000,
                 anneal_iters=2000,
                 prior_delta_min=0.5,
                 prior_delta_max=16.0,
                 kl_clip=50.0):
        super(PointSupervisedVPDLoss, self).__init__()
        self.lambda_center = lambda_center
        self.lambda_kl = lambda_kl
        self.lambda_kl_warmup = lambda_kl_warmup
        self.lambda_var = lambda_var
        self.knn_k = knn_k
        self.sigma_c_coeff = sigma_c_coeff
        self.scale_alpha_w = scale_alpha_w
        self.scale_alpha_h = scale_alpha_h
        self.sigma_s_init = sigma_s_init
        self.sigma_s_final = sigma_s_final
        self.warmup_iters = warmup_iters
        self.anneal_iters = anneal_iters
        self.prior_delta_min = prior_delta_min
        self.prior_delta_max = prior_delta_max
        self.kl_clip = kl_clip

    def _curriculum(self, cur_iter):
        """Return (eff_lambda_kl, sigma_s) for current iteration."""
        if cur_iter < self.warmup_iters:
            return self.lambda_kl_warmup, self.sigma_s_init
        ratio = min(1.0, (cur_iter - self.warmup_iters) / max(self.anneal_iters, 1))
        eff_lambda_kl = self.lambda_kl_warmup + ratio * (self.lambda_kl - self.lambda_kl_warmup)
        sigma_s = self.sigma_s_init - ratio * (self.sigma_s_init - self.sigma_s_final)
        return eff_lambda_kl, sigma_s

    def _compute_di_norm(self, gt_centers_norm, all_gt_centers_norm):
        """Compute mean kNN distance in normalized space for each positive sample.

        Args:
            gt_centers_norm (Tensor): Matched GT center in normalized space (N, 2).
                Format: (gt_center - anchor) / stride
            all_gt_centers_norm (Tensor): All GT centers normalized (M, 2).

        Returns:
            Tensor: d_i per sample (N,), clamped to [prior_delta_min, prior_delta_max].
        """
        N, M = gt_centers_norm.shape[0], all_gt_centers_norm.shape[0]
        if M <= 1:
            return gt_centers_norm.new_full((N,), self.prior_delta_max)

        dists = torch.cdist(gt_centers_norm, all_gt_centers_norm)  # (N, M)
        # Mask self (distance ~0)
        dists = dists + (dists < 1e-2).float() * 1e8
        k = min(self.knn_k, M - 1)
        knn_dists, _ = dists.topk(k, dim=1, largest=False)
        d_i = knn_dists.mean(dim=1)
        return d_i.clamp(min=self.prior_delta_min, max=self.prior_delta_max)

    def forward(self,
                bbox_mu,
                bbox_log_sigma,
                pos_points,
                pos_strides,
                gt_centers,
                gt_centers_list,
                cur_iter=0):
        """Compute point-supervised VPD loss in stride-normalized space.

        Args:
            bbox_mu (Tensor): Posterior mean (N, 4).
                [:, :2] = (delta_x/stride, delta_y/stride)  -- normalized center offset
                [:, 2:] = (log_w, log_h)  -- log size (no stride)
            bbox_log_sigma (Tensor): Posterior log-std (N, 4).
            pos_points (Tensor): Anchor points in image coords (N, 2).
            pos_strides (Tensor): Stride per positive sample (N,).
            gt_centers (Tensor): Matched GT center in image coords (N, 2).
            gt_centers_list (list[Tensor]): All GT centers per image in image coords.
            cur_iter (int): Current training iteration.

        Returns:
            dict[str, Tensor]: loss_center, loss_kl, loss_var, loss_total.
        """
        N = bbox_mu.shape[0]
        if N == 0:
            zero = bbox_mu.sum() * 0.0
            return dict(loss_center=zero, loss_kl=zero, loss_var=zero, loss_total=zero)

        stride_2d = pos_strides.unsqueeze(1)  # (N, 1)

        # --- Center loss in normalized space ---
        # target: (gt_center - anchor) / stride
        gt_delta_norm = (gt_centers - pos_points) / stride_2d  # (N, 2)
        pred_delta_norm = bbox_mu[:, :2]                        # (N, 2)
        l_center = F.smooth_l1_loss(pred_delta_norm, gt_delta_norm,
                                    reduction='mean', beta=1.0)

        # --- Build point-conditioned prior in normalized space ---
        eff_lambda_kl, sigma_s = self._curriculum(cur_iter)

        # For kNN, normalize all GT centers relative to each sample's anchor/stride.
        # Simpler: use pixel-space kNN distances, then divide by mean stride.
        # Concatenate all GT centers (pixel space) for kNN lookup.
        all_gt_centers_px = torch.cat(gt_centers_list, dim=0)  # (M, 2)
        mean_stride = pos_strides.float().mean()

        # kNN distances in pixel space, then normalize
        if all_gt_centers_px.shape[0] > 1:
            dists_px = torch.cdist(gt_centers, all_gt_centers_px)  # (N, M)
            dists_px = dists_px + (dists_px < 1e-2).float() * 1e8
            k = min(self.knn_k, all_gt_centers_px.shape[0] - 1)
            knn_dists_px, _ = dists_px.topk(k, dim=1, largest=False)
            d_i_px = knn_dists_px.mean(dim=1)  # (N,) in pixels
        else:
            d_i_px = gt_centers.new_full((N,), mean_stride * self.prior_delta_max)

        # Normalize d_i by per-sample stride
        d_i_norm = (d_i_px / pos_strides.float()).clamp(
            min=self.prior_delta_min, max=self.prior_delta_max)  # (N,)

        # Center prior: mu=0, sigma = sigma_c_coeff * d_i_norm  (in normalized units)
        sigma_c = (self.sigma_c_coeff * d_i_norm).unsqueeze(1).expand(-1, 2)  # (N, 2)
        mu_c = torch.zeros(N, 2, device=bbox_mu.device)

        # Scale prior: mu = log(alpha * d_i_norm), sigma = sigma_s
        mu_s = torch.stack([
            torch.log(self.scale_alpha_w * d_i_norm),
            torch.log(self.scale_alpha_h * d_i_norm)], dim=1)   # (N, 2)
        sigma_s_2d = bbox_mu.new_full((N, 2), sigma_s)

        prior_mu = torch.cat([mu_c, mu_s], dim=1)              # (N, 4)
        prior_sigma = torch.cat([sigma_c, sigma_s_2d], dim=1)  # (N, 4)

        # --- KL loss with per-sample clipping to prevent spikes ---
        sigma_q = bbox_log_sigma.exp().clamp(min=1e-6)
        sigma_p = prior_sigma.clamp(min=1e-6)
        kl_per_dim = (torch.log(sigma_p / sigma_q)
                      + (sigma_q.pow(2) + (bbox_mu - prior_mu).pow(2))
                      / (2.0 * sigma_p.pow(2))
                      - 0.5)
        kl_per_sample = kl_per_dim.sum(dim=-1)  # (N,)
        # Clip per-sample KL to prevent a few outlier samples from causing divergence
        kl_per_sample = kl_per_sample.clamp(max=self.kl_clip)
        l_kl = kl_per_sample.mean()

        # --- Variance regularization on center dims ---
        l_var = bbox_log_sigma[:, :2].exp().mean()

        loss_total = (self.lambda_center * l_center
                      + eff_lambda_kl * l_kl
                      + self.lambda_var * l_var)

        return dict(
            loss_center=l_center,
            loss_kl=l_kl,
            loss_var=l_var,
            loss_total=loss_total,
        )
