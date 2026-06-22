import torch
import torch.nn as nn
from mmdet.core import reduce_mean

from ..builder import ROTATED_LOSSES

@ROTATED_LOSSES.register_module()
class VariationalClassificationLoss(nn.Module):
    def __init__(self, lambda_cls_var=1.0, project_min=-8.0, project_max=8.0, num_bins=33, eps=1e-6):
        super(VariationalClassificationLoss, self).__init__()
        self.lambda_cls_var = lambda_cls_var
        self.project_min = float(project_min)
        self.project_max = float(project_max)
        self.num_bins = int(num_bins)
        self.eps = float(eps)
        self.log_sigma_min = -7.0
        self.log_sigma_max = 5.0
        self.sigma_max = 1e2
        
        # Ideal logit targets for positive and negative classes
        self.target_logit_pos = 5.0  
        self.target_logit_neg = -5.0 
        
        if self.num_bins < 2:
            raise ValueError('num_bins must be >= 2')
        support = torch.linspace(self.project_min, self.project_max, steps=self.num_bins, dtype=torch.float32)
        self.register_buffer('support', support)
        self.bin_width = (self.project_max - self.project_min) / (self.num_bins - 1)

    def _sanitize_log_sigma(self, log_sigma):
        log_sigma = log_sigma.clamp(min=self.log_sigma_min, max=self.log_sigma_max)
        log_sigma = torch.nan_to_num(log_sigma, nan=0.0, posinf=self.log_sigma_max, neginf=self.log_sigma_min)
        return log_sigma

    def _project_target_dist(self, target):
        """Project scalar target logit values onto linear-interpolated support bins."""
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
        m = 0.5 * (p + q)
        kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=1)
        kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=1)
        return 0.5 * (kl_pm + kl_qm)

    def forward(self, flatten_cls_score, flatten_cls_logvars, flatten_labels):
        """
        Args:
            flatten_cls_score (Tensor): (N, C) predicted classification logits (MUST BE DETACHED before calling)
            flatten_cls_logvars (Tensor): (N, 1) predicted shared log-variance
            flatten_labels (Tensor): (N,) class labels (0 to C-1 for pos, C for bg)
        """
        N, C = flatten_cls_score.shape
        if N == 0:
            return flatten_cls_score.sum() * 0.0

        # 1. Build one-hot style targets in logit space
        with torch.no_grad():
            # Create target logits: pos classes -> 5.0, neg classes -> -5.0
            target_logits = torch.full_like(flatten_cls_score, self.target_logit_neg) # (N, C) init as neg
            pos_mask = torch.zeros_like(flatten_cls_score, dtype=torch.bool)
            valid_pos = (flatten_labels >= 0) & (flatten_labels < C)
            if valid_pos.any():
                pos_idx = flatten_labels[valid_pos].long()
                pos_mask[valid_pos] = pos_mask[valid_pos].scatter(1, pos_idx.unsqueeze(1), True)
            target_logits[pos_mask] = self.target_logit_pos

        # 2. Prepare prediction parameters
        # Detach mu to prevent this loss from affecting CPM's classification branch
        mu = flatten_cls_score.detach() 
        log_sigma = flatten_cls_logvars.expand_as(mu) # (N, C) shared variance
        log_sigma = self._sanitize_log_sigma(log_sigma)
        sigma = log_sigma.exp().clamp(min=1e-6, max=self.sigma_max)

        # 3. Flatten to compute distributions per class per sample
        mu_flat = mu.reshape(-1) # (N*C,)
        sigma_flat = sigma.reshape(-1) # (N*C,)
        target_flat = target_logits.reshape(-1) # (N*C,)

        # 4. Discretize and match
        target_dist = self._project_target_dist(target_flat) # (N*C, K)
        pred_dist = self._gaussian_binned_dist(mu_flat, sigma_flat) # (N*C, K)
        
        js_div = self._js_divergence(pred_dist, target_dist) # (N*C,)
        
        # 5. Average over all samples and classes
        loss = js_div.mean() * self.lambda_cls_var
        
        return loss

        
            
            
        
