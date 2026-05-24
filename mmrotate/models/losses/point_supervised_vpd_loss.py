# Copyright (c) OpenMMLab. All rights reserved.
"""Point-Supervised VPD Loss with Point-Conditioned Prior.

All computations are done in **stride-normalized space**:
    - bbox_mu[:, :2] = (delta_x / stride, delta_y / stride)  [network output]
    - gt_center_delta = (gt_center - anchor) / stride  [target in same space]
    - d_i_norm = d_i_pixels / stride  [kNN distance normalized]

This ensures center loss and KL prior/posterior are all in the same units.

Objective:
    L = lambda_center * L_nll
                + lambda_kl(t) * SymKL(q_phi, p_psi)

Curriculum:
    Stage A (iter < warmup_iters): lambda_kl = lambda_kl_warmup,
                                                                    lambda_var = lambda_var_warmup
  Stage B (iter >= warmup_iters): lambda_kl linearly increases to lambda_kl,
                                                                     lambda_var linearly increases to lambda_var
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES

NAN_TO_NUM = False


@ROTATED_LOSSES.register_module()
class PointSupervisedVPDLoss(nn.Module):
    """Point-Supervised VPD Loss with point-conditioned prior in normalized space.

    Args:
        lambda_center (float): Weight for center regression loss. Default: 1.0.
        lambda_kl (float): Final KL weight (stage B). Default: 0.1.
        lambda_kl_warmup (float): Initial KL weight (stage A). Default: 0.02.
        lambda_var (float): Final variance regularization weight (stage B).
            Default: 0.01. (Deprecated when use_nll=True)
        lambda_var_warmup (float | None): Initial variance regularization
            weight (stage A). If None, uses lambda_var (no schedule).
            Default: None. (Deprecated when use_nll=True)
        use_nll (bool): If True, use heteroscedastic Gaussian NLL for
            reconstruction so sigma learns uncertainty. Default: True.
        knn_k (int): Nearest neighbors for density estimation. Default: 5.
        sigma_c_coeff (float): Center prior sigma = sigma_c_coeff * d_i_norm.
            Default: 0.5.
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
                 lambda_var=0.04,
                 lambda_var_warmup=0.001,
                 num_samples=1,
                 use_nll=True,
                 knn_k=5,
                 sigma_c_coeff=0.5,
                 warmup_iters=2000,
                 anneal_iters=800,
                 prior_delta_min=0.5,
                 prior_delta_max=16.0,
                 kl_clip=50.0):
        super(PointSupervisedVPDLoss, self).__init__()
        self.lambda_center = lambda_center
        self.lambda_kl = lambda_kl
        self.lambda_kl_warmup = lambda_kl_warmup
        self.lambda_var = lambda_var
        self.lambda_var_warmup = (
            lambda_var if lambda_var_warmup is None else lambda_var_warmup)
        self.num_samples = max(int(num_samples), 1)
        self.use_nll = bool(use_nll)
        self.knn_k = knn_k
        self.sigma_c_coeff = sigma_c_coeff
        self.warmup_iters = warmup_iters
        self.anneal_iters = anneal_iters
        self.prior_delta_min = prior_delta_min
        self.prior_delta_max = prior_delta_max
        self.kl_clip = kl_clip
        self.log_sigma_min = -7.0
        self.log_sigma_max = 5.0
        self.sigma_max = 1e2

    def _sanitize_log_sigma(self, bbox_log_sigma):
        bbox_log_sigma = bbox_log_sigma.clamp(
            min=self.log_sigma_min, max=self.log_sigma_max)
        bbox_log_sigma = torch.nan_to_num(
            bbox_log_sigma,
            nan=0.0,
            posinf=self.log_sigma_max,
            neginf=self.log_sigma_min)
        return bbox_log_sigma

    def _curriculum(self, cur_iter):
        """Return (eff_lambda_kl, eff_lambda_var) for current iteration."""
        if cur_iter < self.warmup_iters:
            return self.lambda_kl_warmup, self.lambda_var_warmup
        ratio = min(1.0, (cur_iter - self.warmup_iters) / max(self.anneal_iters, 1))
        eff_lambda_kl = self.lambda_kl_warmup + ratio * (self.lambda_kl - self.lambda_kl_warmup)
        eff_lambda_var = self.lambda_var_warmup + ratio * (self.lambda_var - self.lambda_var_warmup)
        return eff_lambda_kl, eff_lambda_var

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
                cur_iter=0,
                pos_img_ids=None,
                num_samples=None,
                prior_log_sigma=None):
        """Compute point-supervised VPD loss in stride-normalized space.

        Args:
            bbox_mu (Tensor): Posterior mean (N, 2).
                [:, :2] = (delta_x/stride, delta_y/stride)  -- normalized center offset
            bbox_log_sigma (Tensor): Posterior log-std (N, 2).
            pos_points (Tensor): Anchor points in image coords (N, 2).
            pos_strides (Tensor): Stride per positive sample (N,).
            gt_centers (Tensor): Matched GT center in image coords (N, 2).
            gt_centers_list (list[Tensor]): All GT centers per image in image coords.
            prior_log_sigma (Tensor | None): Optional prior log-std from features,
                shape (N, 2). If provided, overrides kNN-based prior.
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

        # --- Reconstruction loss in normalized space ---
        # target: (gt_center - anchor) / stride
        gt_delta_norm = (gt_centers - pos_points) / stride_2d  # (N, 2)
        if self.use_nll:
            # Gaussian NLL: 0.5 * ((x-mu)^2 / sigma^2 + 2*log(sigma))
            log_sigma = bbox_log_sigma
            inv_var = torch.exp(-2.0 * log_sigma)
            diff = gt_delta_norm - bbox_mu
            l_center = 0.5 * (diff.pow(2) * inv_var + 2.0 * log_sigma).mean()
        else:
            # Monte Carlo estimate of E[recon] with sampled z
            num_samples = self.num_samples if num_samples is None else max(int(num_samples), 1)
            eps = torch.randn(
                num_samples, N, 2, device=bbox_mu.device, dtype=bbox_mu.dtype)
            sample_delta_norm = bbox_mu.unsqueeze(0) + sigma_q.unsqueeze(0) * eps
            target_delta_norm = gt_delta_norm.unsqueeze(0).expand_as(sample_delta_norm)
            l_center = F.smooth_l1_loss(
                sample_delta_norm,
                target_delta_norm,
                reduction='none',
                beta=1.0).mean()
        if NAN_TO_NUM:
            l_center = torch.nan_to_num(l_center, nan=0.0, posinf=1e4, neginf=0.0)

        # --- Build point-conditioned prior in normalized space ---
        eff_lambda_kl, eff_lambda_var = self._curriculum(cur_iter)

        prior_mu = torch.zeros(N, 2, device=bbox_mu.device)
        if prior_log_sigma is not None:
            prior_log_sigma = self._sanitize_log_sigma(prior_log_sigma)
            prior_sigma = prior_log_sigma.exp().clamp(min=1e-6, max=self.sigma_max)
            if NAN_TO_NUM:
                prior_sigma = torch.nan_to_num(
                    prior_sigma, nan=1.0, posinf=self.sigma_max, neginf=1e-6)
        else:
            if pos_img_ids is not None:
                d_i_norm = pos_strides.new_full((N,), self.prior_delta_max)
                for img_idx, gt_centers_img in enumerate(gt_centers_list):
                    mask = (pos_img_ids == img_idx)
                    if not mask.any():
                        continue
                    if gt_centers_img.shape[0] <= 1:
                        d_i_px = gt_centers.new_full(
                            (int(mask.sum()),),
                            pos_strides[mask].float().mean() * self.prior_delta_max)
                    else:
                        dists_px = torch.cdist(gt_centers[mask], gt_centers_img)  # (n_i, m_i)
                        dists_px = dists_px + (dists_px < 1e-2).float() * 1e8
                        k = min(self.knn_k, gt_centers_img.shape[0] - 1)
                        knn_dists_px, _ = dists_px.topk(k, dim=1, largest=False)
                        d_i_px = knn_dists_px.mean(dim=1)
                    d_i_norm[mask] = (d_i_px / pos_strides[mask].float()).clamp(
                        min=self.prior_delta_min, max=self.prior_delta_max)
            else:
                # Fallback: use batch-level GT centers for kNN
                all_gt_centers_px = torch.cat(gt_centers_list, dim=0)  # (M, 2)
                mean_stride = pos_strides.float().mean()
                if all_gt_centers_px.shape[0] > 1:
                    dists_px = torch.cdist(gt_centers, all_gt_centers_px)  # (N, M)
                    dists_px = dists_px + (dists_px < 1e-2).float() * 1e8
                    k = min(self.knn_k, all_gt_centers_px.shape[0] - 1)
                    knn_dists_px, _ = dists_px.topk(k, dim=1, largest=False)
                    d_i_px = knn_dists_px.mean(dim=1)  # (N,) in pixels
                else:
                    d_i_px = gt_centers.new_full((N,), mean_stride * self.prior_delta_max)
                d_i_norm = (d_i_px / pos_strides.float()).clamp(
                    min=self.prior_delta_min, max=self.prior_delta_max)  # (N,)

            # Center prior: mu=0, sigma = sigma_c_coeff * d_i_norm  (in normalized units)
            prior_sigma = (self.sigma_c_coeff * d_i_norm).unsqueeze(1).expand(-1, 2)  # (N, 2)
            if NAN_TO_NUM:
                prior_sigma = torch.nan_to_num(
                    prior_sigma, nan=1.0, posinf=self.sigma_max, neginf=1e-6)

        if NAN_TO_NUM:
            prior_mu = torch.nan_to_num(prior_mu, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- Symmetric KL loss with per-sample clipping to prevent spikes ---
        sigma_p = prior_sigma.clamp(min=1e-6, max=self.sigma_max)
        if NAN_TO_NUM:
            sigma_p = torch.nan_to_num(sigma_p, nan=1.0, posinf=self.sigma_max, neginf=1e-6)

        delta_sq = (bbox_mu - prior_mu).pow(2)
        kl_qp_per_dim = (
            torch.log(sigma_p / sigma_q)
            + (sigma_q.pow(2) + delta_sq) / (2.0 * sigma_p.pow(2))
            - 0.5)
        kl_pq_per_dim = (
            torch.log(sigma_q / sigma_p)
            + (sigma_p.pow(2) + delta_sq) / (2.0 * sigma_q.pow(2))
            - 0.5)
        sym_kl_per_dim = 0.5 * (kl_qp_per_dim + kl_pq_per_dim)
        if NAN_TO_NUM:
            sym_kl_per_dim = torch.nan_to_num(sym_kl_per_dim, nan=0.0, posinf=1e4, neginf=0.0)

        kl_per_sample = sym_kl_per_dim.sum(dim=-1)  # (N,)
        # Clip per-sample KL to prevent a few outlier samples from causing divergence
        kl_per_sample = kl_per_sample.clamp(max=self.kl_clip)
        if NAN_TO_NUM:
            kl_per_sample = torch.nan_to_num(kl_per_sample, nan=0.0, posinf=self.kl_clip, neginf=0.0)
        l_kl = kl_per_sample.mean()
        if NAN_TO_NUM:
            l_kl = torch.nan_to_num(l_kl, nan=0.0, posinf=self.kl_clip, neginf=0.0)

        # --- Variance regularization on center dims (only when not using NLL) ---
        if self.use_nll:
            l_var = bbox_mu.sum() * 0.0
        else:
            l_var = bbox_log_sigma[:, :2].exp().clamp(max=self.sigma_max).mean()
            if NAN_TO_NUM:
                l_var = torch.nan_to_num(l_var, nan=0.0, posinf=self.sigma_max, neginf=0.0)

        loss_total = (self.lambda_center * l_center
                      + eff_lambda_kl * l_kl
                      + eff_lambda_var * l_var)
        if NAN_TO_NUM:
            loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=0.0)

        return dict(
            loss_center=l_center,
            loss_kl=l_kl,
            loss_var=l_var,
            loss_total=loss_total,
        )
