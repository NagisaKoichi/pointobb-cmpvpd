import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ..builder import ROTATED_LOSSES

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
        mu_pred = torch.sigmoid(vi_mu_logit) # Mean in [0, 1]
        
        # Temperature scaling affects the sharpness
        # sigma_pred = exp(log_sigma) / temperature
        sigma_pred = (vi_log_sigma.exp() / max(temperature, 1e-6)).clamp(min=self.eps)

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
        # A. Reconstruction Loss (Mean L2) - Strong gradient for correct mean
        loss_mu_l2 = (mu_pred - mu_target) ** 2
        loss_mu_pos = (loss_mu_l2 * pos_mask).sum() / num_pos
        loss_mu_neg = (loss_mu_l2 * neg_mask).sum() / num_neg
        
        # or use soft_labels to weight the loss after divided
        # loss_mu_pos = (loss_mu_l2 * pos_mask * soft_labels).sum() / num_pos
        # loss_mu_neg = (loss_mu_l2 * neg_mask * (1.0 - soft_labels)).sum() / num_neg
        
        # loss_recon_mu = 0.5 * loss_mu_pos + 0.5 * loss_mu_neg
        loss_recon_mu = 0.0 * loss_mu_pos + 0.0 * loss_mu_neg  # disable loss_recon_mu

        # B. Distribution Matching (KL Divergence) - Supervise Sigma/Uncertainty
        # KL(Pred || Target)
        kl_val = self._kl_divergence_gaussian(mu_pred, sigma_pred, mu_target, sigma_target)
        
        loss_kl_pos = (kl_val * pos_mask).sum() / num_pos
        loss_kl_neg = (kl_val * neg_mask).sum() / num_neg
        loss_dist_match = 0.5 * loss_kl_pos + 0.5 * loss_kl_neg

        # C. Regularization (Prevent sigma from becoming too small or too large)
        # L2 on log_sigma encourages it to stay around 0 (sigma=1), which is moderate.
        # Or simply penalize very small sigma to prevent numerical issues.
        loss_reg_sigma = (vi_log_sigma ** 2).mean()
        
        # alternatively, use js divergence with a gaussian distribution.
        # C: Regularization: use js divergence with a gaussian distribution with mean=0.5, sigma=1.0

        # Total Loss
        # Weight: 1.0 for Mean (accuracy), 1.0 for KL (uncertainty shape), small weight for reg
        loss_total = self.lambda_vi * (1.0 * loss_recon_mu + 0.01 * loss_dist_match) + 0.01 * loss_reg_sigma
        
        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=-1e4)

        return dict(
            loss_vi=loss_recon_mu,       # Monitor mean accuracy
            loss_vi_kl=0.01*loss_dist_match,  # Monitor uncertainty learning
            loss_total=loss_total,
            vi_prob=mu_pred.detach()
        )
