import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ..builder import ROTATED_LOSSES

lambda_dist_match = 0.01
lambda_sampled = 0.1
lambda_reg = 0.01
sigma_scale = 1.0

@ROTATED_LOSSES.register_module()
class VIPosNegLoss(nn.Module):
    def __init__(self, lambda_vi=1.0, base_sigma=0.05, max_sigma=0.5, eps=1e-6, num_bins=21, **kwargs):
        """
        Args:
            lambda_vi (float): Loss weight.
            base_sigma (float): Minimum target sigma for confident regions (center/bg).
            max_sigma (float): Maximum target sigma for uncertain regions (edge).
            eps (float): Numerical stability term.
            num_bins (int): Number of bins for discretizing the distribution [0, 1].
        """
        super(VIPosNegLoss, self).__init__()
        self.lambda_vi = lambda_vi
        self.eps = eps
        self.base_sigma = base_sigma
        self.max_sigma = max_sigma
        self.num_bins = num_bins
        # Register bins centers for discretization
        self.register_buffer('prob_bins', torch.linspace(0, 1, num_bins))

    def _sanitize_log_sigma(self, log_sigma):
        # Clamp log_sigma to prevent explosion.
        return log_sigma.clamp(min=-5.0, max=3.0)

    def _generate_soft_labels(self, dist_to_gt, pos_thresh):
        """ Generate soft labels based on distance to ground truth. """
        thresh = pos_thresh.clamp(min=1.0)
        normalized_dist = dist_to_gt / thresh
        soft = torch.exp(-0.5 * (normalized_dist / 0.4) ** 2)
        return soft.clamp(min=0.0, max=1.0)

    def _generate_target_gaussian_params(self, soft_labels):
        """ Generate target Mean and Sigma. """
        mu_target = soft_labels
        dist_to_boundary = torch.abs(mu_target - 0.5)
        certainty = 2.0 * dist_to_boundary
        sigma_target = self.base_sigma + (1.0 - certainty) * (self.max_sigma - self.base_sigma)
        return mu_target, sigma_target

    def _get_discrete_distribution(self, mu, sigma):
        """
        Discretize Gaussian parameters into a probability mass function over bins.
        Args:
            mu (Tensor): Mean of the distribution (N,).
            sigma (Tensor): Std of the distribution (N,).
        Returns:
            Tensor: Discrete distribution (N, num_bins).
        """
        # bins: (1, num_bins)
        bins = self.prob_bins.unsqueeze(0).to(mu.device)
        # mu: (N, 1), sigma: (N, 1)
        mu = mu.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
        
        # Calculate PDF values at bin centers (unnormalized)
        # dist ~ exp(-0.5 * (bins - mu)^2 / sigma^2)
        dist = torch.exp(-0.5 * ((bins - mu) / sigma)**2)
        
        # Normalize to sum to 1 (PMF)
        dist = dist / (dist.sum(dim=-1, keepdim=True) + self.eps)
        return dist

    def _js_divergence_discrete(self, p, q):
        """
        Compute JS divergence between two discrete distributions.
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
        """
        m = 0.5 * (p + q)
        kl_pm = (p * torch.log(p / (m + self.eps) + self.eps)).sum(dim=-1)
        kl_qm = (q * torch.log(q / (m + self.eps) + self.eps)).sum(dim=-1)
        js = 0.5 * (kl_pm + kl_qm)
        return js

    def forward(self, vi_mu_logit, vi_log_sigma, dist_to_gt, pos_thresh, is_ignored=None, is_pos=None, temperature=1.0, vi_target=None, iter=-1):
        N = vi_mu_logit.shape[0]
        if N == 0:
            zero = vi_mu_logit.sum() * 0.0
            return dict(loss_vi=zero, loss_total=zero, vi_prob=zero.detach())

        # 1. Sanitize and get predictions
        vi_log_sigma = self._sanitize_log_sigma(vi_log_sigma)
        mu_pred = torch.sigmoid(vi_mu_logit)
        sigma_pred = vi_log_sigma.exp().clamp(min=self.eps)

        # 2. Generate Targets
        if vi_target is None:
            soft_labels = self._generate_soft_labels(dist_to_gt, pos_thresh)
        else:
            soft_labels = vi_target
        mu_target, sigma_target = self._generate_target_gaussian_params(soft_labels)

        # 3. Masking
        if is_pos is not None:
            pos_mask = is_pos.float()
            neg_mask = (1.0 - pos_mask) * (~is_ignored).float() if is_ignored is not None else (1.0 - pos_mask)
        else:
            pos_mask = (soft_labels > 0.5).float()
            neg_mask = (soft_labels <= 0.5).float()
        num_pos = pos_mask.sum().clamp(min=1.0)
        num_neg = neg_mask.sum().clamp(min=1.0)

        # 4. Calculate Loss - Discrete Distribution Matching
        # Discretize predicted and target Gaussians
        pred_dist = self._get_discrete_distribution(mu_pred, sigma_pred)
        target_dist = self._get_discrete_distribution(mu_target, sigma_target)
        
        # JS Divergence Loss
        js_val = self._js_divergence_discrete(pred_dist, target_dist)
        loss_kl_pos = (js_val * pos_mask).sum() / num_pos
        loss_kl_neg = (js_val * neg_mask).sum() / num_neg
        loss_reg = (0.5 * loss_kl_pos + 0.5 * loss_kl_neg) * lambda_reg

        # 5. Sampled Matching (Logit space)
        target_logits = torch.log(mu_target / (1.0 - mu_target + 1e-6) + 1e-6)
        eps = torch.randn_like(vi_mu_logit)
        sample_logits = vi_mu_logit + vi_log_sigma.exp() * eps
        
        # loss_sampled_pos = (torch.abs(sample_logits - target_logits) * pos_mask).sum() / num_pos
        # loss_sampled_neg = (torch.abs(sample_logits - target_logits) * neg_mask).sum() / num_neg
        
        # or use smoothl1
        loss_sampled_all = F.smooth_l1_loss(sample_logits, target_logits, reduction='none')
        loss_sampled_pos = (loss_sampled_all * pos_mask).sum() / num_pos
        loss_sampled_neg = (loss_sampled_all * neg_mask).sum() / num_neg
        
        loss_sampled = (0.5 * loss_sampled_pos + 0.5 * loss_sampled_neg) * lambda_sampled

        # 6. Regularization (Discrete JS vs Uniform)
        # Regularize against uniform distribution to prevent degenerate solutions
        # uniform_dist = torch.ones_like(pred_dist) / self.num_bins
        # loss_reg_sigma = self._js_divergence_discrete(pred_dist, uniform_dist).mean() * lambda_reg

        # Total Loss
        # loss_total = self.lambda_vi * (loss_sampled + loss_dist_match) + loss_reg_sigma
        loss_total = self.lambda_vi * (loss_sampled) + loss_reg
        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=-1e4)

        return dict(
            loss_vi_sampled=loss_sampled,
            loss_vi_reg=loss_reg,
            loss_total=loss_total,
            vi_prob=mu_pred.detach(),
            vi_sigma=sigma_pred.detach()
        )
