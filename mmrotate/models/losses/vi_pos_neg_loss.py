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
    def __init__(self, lambda_vi=1.0, base_sigma=0.05, max_sigma=0.5, eps=1e-6, **kwargs):
        """
        Args:
            lambda_vi (float): Loss weight.
            base_sigma (float): Minimum target sigma for confident regions (center/bg).
            max_sigma (float): Maximum target sigma for uncertain regions (edge).
            eps (float): Numerical stability term.
        """
        super(VIPosNegLoss, self).__init__()
        self.lambda_vi = lambda_vi
        self.eps = eps
        self.base_sigma = base_sigma
        self.max_sigma = max_sigma

    def _sanitize_log_sigma(self, log_sigma):
        # Clamp log_sigma to prevent explosion. 
        # sigma range approx [0.01, 10] -> log range [-4.6, 2.3]
        return log_sigma.clamp(min=-5.0, max=3.0)

    def _generate_soft_labels(self, dist_to_gt, pos_thresh):
        """
        Generate soft labels based on distance to ground truth and positive threshold.
        Logic:
        - If distance < pos_thresh: soft_label = 1.0 (positive)
        - If distance > pos_thresh: soft_label decays to 0.0 (negative)
            - decay by Gaussian function: exp(-0.5 * (dist/sigma*thresh)^2) with fixed sigma
        - Use a Gaussian decay for smooth transition.
        """
        
        # Same Gaussian decay as before
        thresh = pos_thresh.clamp(min=1.0)
        normalized_dist = dist_to_gt / thresh
        # soft = torch.exp(-0.5 * (normalized_dist / 0.4) ** 2) # sigma fixed to 0.4 for label shape
        soft = torch.exp(-0.5 * (normalized_dist / 0.4) ** 2) # sigma fixed to 0.4 for label shape
        return soft.clamp(min=0.0, max=1.0)

    def _generate_target_gaussian_params(self, soft_labels):
        """
        Generate target Mean and Sigma for Gaussian distribution matching.
        Logic:
        - Target Mean = soft_labels (0~1)
        - Target Sigma: 
          - If soft_label close to 0 or 1 (Confident) -> sigma = base_sigma (small)
          - If soft_label close to 0.5 (Uncertain)   -> sigma = max_sigma (large)
        """
        mu_target = soft_labels
        
        # Calculate distance to decision boundary (0.5)
        # distance = |mu_t - 0.5|. Range [0, 0.5]
        dist_to_boundary = torch.abs(mu_target - 0.5)
        
        # Normalize to [0, 1] where 0=Edge(Uncertain), 1=Center(Confident)
        # certainty = 2 * dist_to_boundary
        certainty = 2.0 * dist_to_boundary
        
        # Map certainty to sigma range
        # High certainty -> Low sigma. Low certainty -> High sigma.
        # sigma_t = base + (1 - certainty) * (max - base)
        sigma_target = self.base_sigma + (1.0 - certainty) * (self.max_sigma - self.base_sigma)
        
        return mu_target, sigma_target

    def _kl_divergence_gaussian(self, mu_p, sigma_p, mu_q, sigma_q):
        """
        Compute KL(p || q) between two Gaussians.
        KL(p||q) = log(sigma_q/sigma_p) + (sigma_p^2 + (mu_p-mu_q)^2) / (2*sigma_q^2) - 0.5
        """
        # Ensure numerical stability
        sigma_p = sigma_p.clamp(min=self.eps)
        sigma_q = sigma_q.clamp(min=self.eps)
        
        term1 = torch.log(sigma_q / sigma_p)
        term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
        kl = term1 + term2 - 0.5
        
        # Clamp KL to prevent extreme values
        return kl.clamp(max=10.0)
    
    def _js_divergence_gaussian(self, mu_p, sigma_p, mu_q, sigma_q):
        """
        Compute JS divergence between two Gaussians.
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
        """
        # Compute M parameters
        mu_m = 0.5 * (mu_p + mu_q)
        sigma_m = 0.5 * (sigma_p + sigma_q)
        
        kl_pm = self._kl_divergence_gaussian(mu_p, sigma_p, mu_m, sigma_m)
        kl_qm = self._kl_divergence_gaussian(mu_q, sigma_q, mu_m, sigma_m)
        
        js = 0.5 * (kl_pm + kl_qm)
        
        return js.clamp(max=10.0)
    
    def _visualize_vi_predictions(self, soft_labels, pos_mask, save_path):
        """Visualize VI predictions.

        Args:
            vi_preds (Tensor): VI predictions (2, H_feat, W_feat).
            save_path (str): Path to save the visualization.
        """
        from PIL import Image
        import numpy as np
        vis_img1 = (soft_labels * 255).cpu().numpy().astype(np.uint8)
        vis_img2 = (pos_mask.float() * 255).cpu().numpy().astype(np.uint8)
        vis_img = np.stack([vis_img1, vis_img2, np.zeros_like(vis_img1)], axis=-1)  # RGB
        print(vis_img.shape)
        
        Image.fromarray(vis_img).save(save_path)


    def forward(self, vi_mu_logit, vi_log_sigma, dist_to_gt, pos_thresh, is_ignored=None, is_pos=None, temperature=1.0, vi_target=None, iter=-1):
        N = vi_mu_logit.shape[0]
        if N == 0:
            zero = vi_mu_logit.sum() * 0.0
            return dict(loss_vi=zero, loss_total=zero, vi_prob=zero.detach())

        # 1. Sanitize and get predictions
        vi_log_sigma = self._sanitize_log_sigma(vi_log_sigma)
        mu_pred = torch.sigmoid(vi_mu_logit) # vi_mu_logit is in range (-inf, +inf), mu_pred in range (0, 1)
        
        # Temperature scaling affects the sharpness
        # sigma_pred = exp(log_sigma) / temperature
        # sigma_pred = (vi_log_sigma.exp() / max(temperature, 1e-6)).clamp(min=self.eps)  # 这样在冻结训练时，会导致 sigma_pred 过大
        sigma_pred = (vi_log_sigma.exp() * max(temperature, 1e-6)).clamp(min=self.eps)  # 改成随着 temperature 减小，sigma_pred 减小，鼓励模型在后期更自信

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
        
        # if iter >= 0:
        #     self._visualize_vi_predictions(mu_target.detach().cpu(), sigma_target.detach().cpu(), f"/media/ps/passport2/zlk/results/0627_posnegvi_gaussian_norecon/vpd_cpm_dotav10/soft_labels/soft_labels_iter_{iter}.png")

        # 4. Calculate Loss

        # B. Distribution Matching (KL Divergence) - Supervise Sigma/Uncertainty
        # KL(Pred || Target)
        kl_val = self._js_divergence_gaussian(mu_pred, sigma_pred, mu_target, sigma_target)
        
        loss_kl_pos = (kl_val * pos_mask).sum() / num_pos
        loss_kl_neg = (kl_val * neg_mask).sum() / num_neg
        loss_dist_match = (0.5 * loss_kl_pos + 0.5 * loss_kl_neg) * lambda_dist_match

        # # B. Sampled Matching - Sample from predicted distribution N(mu_pred, sigma_pred)
        # #                       calculate loss with mu_target
        # sample_pred = torch.normal(vi_mu_logit, vi_log_sigma.exp() * sigma_scale).sigmoid()  # Sample from predicted distribution and apply sigmoid to get in range (0, 1)
        # loss_sampled_pos = ((sample_pred - mu_target) ** 2 * pos_mask).sum() / num_pos
        # loss_sampled_neg = ((sample_pred - mu_target) ** 2 * neg_mask).sum() / num_neg
        # loss_sampled = (0.5 * loss_sampled_pos + 0.5 * loss_sampled_neg) * lambda_sampled
        
        # 1. 将 Target 映射回 Logit 空间 (Inverse Sigmoid)  [0.268]
        # 加 eps 防止 log(0)
        target_logits = torch.log(mu_target / (1.0 - mu_target + 1e-6) + 1e-6)

        # 2. 直接在 Logit 空间计算 L1 Loss (MSE也可以，但L1更鲁棒)
        # sample_logits 就是 torch.normal(...) 的结果
        # sample_logits = torch.normal(vi_mu_logit, vi_log_sigma.exp() * sigma_scale)
        eps = torch.randn_like(vi_mu_logit)
        sample_logits = vi_mu_logit + vi_log_sigma.exp() * eps   # 梯度同时流向 mu 和 sigma

        loss_sampled_pos = (torch.abs(sample_logits - target_logits) * pos_mask).sum() / num_pos
        loss_sampled_neg = (torch.abs(sample_logits - target_logits) * neg_mask).sum() / num_neg
        
        # or calculate loss in probability space (after sigmoid)
        # sample_prob = torch.sigmoid(sample_logits)
        # loss_sampled_pos = ((sample_prob - mu_target) ** 2 * pos_mask).sum() / num_pos
        # loss_sampled_neg = ((sample_prob - mu_target) ** 2 * neg_mask).sum() / num_neg
        
        loss_sampled = (0.5 * loss_sampled_pos + 0.5 * loss_sampled_neg) * lambda_sampled

        # C. Regularization (Prevent sigma from becoming too small or too large)
        # L2 on log_sigma encourages it to stay around 0 (sigma=1), which is moderate.
        # Or simply penalize very small sigma to prevent numerical issues.
        loss_reg_sigma = (vi_log_sigma ** 2).mean() * lambda_reg
        
        # alternatively, use js divergence with a gaussian distribution.
        # C: Regularization: use js divergence with a gaussian distribution with mean=0.5, sigma=1.0
        # loss_reg_sigma = self._js_divergence_gaussian(mu_pred, sigma_pred, torch.full_like(mu_pred, 0.5), torch.ones_like(sigma_pred)).mean() * lambda_reg

        # Total Loss
        # Weight: 1.0 for Mean (accuracy), 1.0 for KL (uncertainty shape), small weight for reg
        # loss_total = self.lambda_vi * (0.5 * loss_sampled + 0.5 * loss_dist_match) + loss_reg_sigma
        loss_total = self.lambda_vi * (loss_sampled) + loss_reg_sigma
        
        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=-1e4)

        return dict(
            loss_vi_dist_match=loss_dist_match,       # Monitor mean accuracy
            loss_vi_sampled=loss_sampled,  # Monitor uncertainty learning
            loss_vi_reg=loss_reg_sigma,
            loss_total=loss_total,
            vi_prob=mu_pred.detach(),
            vi_sigma=sigma_pred.detach()
        )