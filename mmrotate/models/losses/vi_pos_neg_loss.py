# Copyright (c) OpenMMLab. All rights reserved.
"""VI Pos/Neg Loss: Variational distribution matching for sample assignment.

Models P(positive) at each location as a Beta distribution, matched to a
distance-based soft-label target distribution via Jensen-Shannon divergence.

Key components:
- Soft label: Gaussian decay from GT center, normalized by assignment threshold
- Predicted distribution: Beta(alpha, beta) discretized onto [0, 1] support bins
- Matching loss: JS divergence between target and predicted binned distributions
"""

import torch
import torch.nn as nn
from torch.distributions import Beta
from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class VIPosNegLoss(nn.Module):
    """Variational Inference Loss for positive/negative sample probability.

    Instead of hard binary label assignment, this loss trains a Beta
    distribution over P(positive) at each location, capturing assignment
    uncertainty near the decision boundary.

    Args:
        lambda_vi (float): Weight for VI matching loss. Default: 1.0.
        lambda_kl (float): Weight for KL-to-prior regularization. Default: 0.01.
        project_min (float): Minimum support value (always 0.0). Default: 0.0.
        project_max (float): Maximum support value (always 1.0). Default: 1.0.
        num_bins (int): Number of support bins for distribution discretization.
        eps (float): Numerical stability epsilon.
        soft_label_sigma (float): Controls soft-label sharpness. Smaller =
            sharper (closer to hard labels). Default: 0.4.
        kappa_min (float): Minimum concentration for numerical stability.
        kappa_max (float): Maximum concentration.
    """

    def __init__(self, lambda_vi=1.0, project_min=0.0, project_max=1.0,
                 num_bins=21, eps=1e-6, soft_label_sigma=0.4,
                 kappa_min=1e-3, kappa_max=1e3, **kwargs):
        super(VIPosNegLoss, self).__init__()
        self.lambda_vi = lambda_vi
        self.project_min = float(project_min)
        self.project_max = float(project_max)
        self.num_bins = int(num_bins)
        self.eps = float(eps)
        self.soft_label_sigma = float(soft_label_sigma)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)

        if self.num_bins < 2:
            raise ValueError('num_bins must be >= 2 for VI distribution.')

        support = torch.linspace(
            self.project_min, self.project_max, steps=self.num_bins,
            dtype=torch.float32)
        self.register_buffer('support', support)
        self.bin_width = (self.project_max - self.project_min) / (self.num_bins - 1)
        
    def _sanitize_log_kappa(self, log_kappa):
        """Clamp log_kappa and replace NaN/inf."""
        log_kappa = log_kappa.clamp(min=-7.0, max=7.0)
        log_kappa = torch.nan_to_num(
            log_kappa, nan=0.0, posinf=7.0, neginf=-7.0)
        return log_kappa

    def _generate_soft_labels(self, dist_to_gt, pos_thresh):
        """Generate soft labels based on distance to nearest GT center.

        Soft label = exp(-0.5 * (d / (sigma * thresh))^2)
        - d=0 (at GT center) → soft_label=1.0 (positive)
        - d=thresh (at boundary) → soft_label=exp(-0.5/sigma^2) ≈ small
        - d >> thresh → soft_label≈0.0 (negative)

        Args:
            dist_to_gt (Tensor): (N,) distance to nearest GT center.
            pos_thresh (Tensor): (N,) positive assignment threshold per point.

        Returns:
            Tensor: (N,) soft labels in [0, 1].
        """
        # Avoid division by zero
        thresh = pos_thresh.clamp(min=1.0)
        normalized_dist = dist_to_gt / thresh
        soft = torch.exp(-0.5 * (normalized_dist / self.soft_label_sigma) ** 2)
        return soft.clamp(min=0.0, max=1.0)

    def _project_target_dist(self, target):
        """Project scalar target in [0,1] onto linear-interpolated support bins.

        Identical interpolation scheme as VPD's _project_target_dist,
        but support is [0, 1] instead of [-16, 16].

        Args:
            target (Tensor): (N,) scalar soft-label target.

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

    def _beta_binned_dist(self, alpha, beta):
        """Discretize Beta(alpha, beta) into support bins via PDF approximation.
        
        PyTorch's Beta distribution does not implement CDF in many versions.
        We compute the PDF at support points and normalize them as the 
        discrete probability mass.
        """
        k = self.num_bins
        support = self.support.to(alpha.dtype)  # (K,)
        
        # Clamp parameters for numerical stability
        alpha = alpha.clamp(min=self.kappa_min, max=self.kappa_max)  # (N,)
        beta = beta.clamp(min=self.kappa_min, max=self.kappa_max)    # (N,)
        
        # Support points must be strictly in (0, 1) for Beta PDF log computation
        x = support.clamp(min=1e-6, max=1.0 - 1e-6)  # (K,)
        
        # Calculate log Beta function: log B(alpha, beta)
        # log B(α, β) = lgamma(α) + lgamma(β) - lgamma(α + β)
        log_beta_fn = (
            torch.lgamma(alpha) + torch.lgamma(beta) 
            - torch.lgamma(alpha + beta)
        )  # (N,)
        
        # Calculate log PDF:
        # log f(x; α, β) = (α - 1) * log(x) + (β - 1) * log(1 - x) - log B(α, β)
        # Broadcasting: (N, 1) * (1, K) - (N, 1) -> (N, K)
        log_pdf = (alpha.unsqueeze(1) - 1.0) * torch.log(x).unsqueeze(0) + \
                  (beta.unsqueeze(1) - 1.0) * torch.log(1.0 - x).unsqueeze(0) - \
                  log_beta_fn.unsqueeze(1)
                  
        # Convert to PDF and normalize to get discrete probabilities
        pdf = torch.exp(log_pdf)
        pred_dist = pdf.clamp(min=self.eps)
        pred_dist = pred_dist / pred_dist.sum(dim=1, keepdim=True)
        
        return pred_dist

    def _js_divergence(self, p, q):
        """Jensen-Shannon divergence for row-wise distributions."""
        m = 0.5 * (p + q)
        kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=1)
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=1)
        return 0.5 * (kl_pm + kl_qm)

    def _kl_to_uniform_prior(self, alpha, beta):
        """KL(Beta(α,β) || Beta(1,1)) = KL(Beta(α,β) || Uniform).

        This regularizes the distribution toward a uniform prior when
        no supervision signal is available (e.g., for negative samples
        far from any GT).

        KL = log B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)

        where B is the Beta function and ψ is the digamma function.
        """
        # Use torch.lgamma for log Beta function
        log_beta_fn = (
            torch.lgamma(alpha) + torch.lgamma(beta)
            - torch.lgamma(alpha + beta))
        digamma_alpha = torch.digamma(alpha)
        digamma_beta = torch.digamma(beta)
        digamma_sum = torch.digamma(alpha + beta)

        kl = (log_beta_fn
              - (alpha - 1) * digamma_alpha
              - (beta - 1) * digamma_beta
              + (alpha + beta - 2) * digamma_sum)
        return kl

    # 修改 forward，加入 temperature 参数
    def forward(self, vi_mu_logit, vi_log_kappa, dist_to_gt, pos_thresh,
                is_ignored=None, temperature=1.0):
        """Compute VI loss.
        
        Args:
            temperature (float): Temperature to control distribution sharpness.
                                 Smaller T leads to sharper (more certain) distribution.
        """
        N = vi_mu_logit.shape[0]
        if N == 0:
            zero = vi_mu_logit.sum() * 0.0
            return dict(loss_vi=zero, loss_total=zero, vi_prob=zero.detach())

        vi_log_kappa = self._sanitize_log_kappa(vi_log_kappa)

        mu = torch.sigmoid(vi_mu_logit)
        # 引入 Temperature: T 越小，kappa 越大，分布越尖锐
        kappa = vi_log_kappa.exp() / max(temperature, 1e-6)
        
        alpha = (mu * kappa).clamp(self.kappa_min, self.kappa_max)
        beta_param = ((1.0 - mu) * kappa).clamp(self.kappa_min, self.kappa_max)

        soft_labels = self._generate_soft_labels(dist_to_gt, pos_thresh)
        target_dist = self._project_target_dist(soft_labels)
        pred_dist = self._beta_binned_dist(alpha, beta_param)

        js = self._js_divergence(target_dist, pred_dist)

        if is_ignored is not None:
            valid_mask = (~is_ignored).float()
            num_valid = valid_mask.sum().clamp(min=1.0)
            loss_vi = (js * valid_mask).sum() / num_valid
        else:
            loss_vi = js.mean()

        loss_total = self.lambda_vi * loss_vi

        return dict(
            loss_vi=loss_vi,
            loss_total=loss_total,
            vi_prob=mu.detach(),
        )