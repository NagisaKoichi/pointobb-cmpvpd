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
        soft = torch.exp(-0.5 * (normalized_dist / 0.8) ** 2) # sigma fixed to 0.4 for label shape
        return soft.clamp(min=0.0, max=1.0)

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

    def _generate_target_gaussian_params(self, soft_labels):
        """
        生成 Logit 空间的目标高斯分布参数。
        
        Args:
            soft_labels (Tensor): [0, 1] 之间的软标签
            
        Returns:
            mu_target (Tensor): 目标均值，范围 
            sigma_target (Tensor): 目标标准差，范围 [base_sigma, max_sigma]
        """
        # 1. 构造 mu_target (Logit 空间)
        # 将 [0, 1] 的概率映射回 (-inf, +inf) 的 Logit 空间
        # 这样 vi_mu_logit 就可以直接在 Logit 空间回归这个值
        soft_labels_clamped = soft_labels.clamp(min=1e-5, max=1.0 - 1e-5)
        mu_target = torch.log(soft_labels_clamped / (1.0 - soft_labels_clamped))
        
        # 2. 构造 sigma_target
        # 逻辑：距离决策边界(0.5)越远，确定性越高，sigma 越小；反之亦然
        dist_to_boundary = torch.abs(soft_labels - 0.5) * 2.0  # range: [0, 1], 1=Very Certain, 0=Very Uncertain
        
        # 映射：High Certainty -> Low Sigma
        sigma_target = self.base_sigma + (1.0 - dist_to_boundary) * (self.max_sigma - self.base_sigma)
        
        return mu_target, sigma_target

    def forward(self, vi_mu_logit, vi_log_sigma, dist_to_gt, pos_thresh, is_ignored=None, is_pos=None, temperature=1.0, vi_target=None, iter=-1):
        N = vi_mu_logit.shape[0]
        if N == 0:
            zero = vi_mu_logit.sum() * 0.0
            return dict(loss_vi=zero, loss_total=zero, vi_prob=zero.detach())

        # ------------------------------------------------------------------
        # 1. 预测分布处理 P(Delta)
        # ------------------------------------------------------------------
        vi_log_sigma = self._sanitize_log_sigma(vi_log_sigma)
        # 预测的均值，作为 Logit 距离度量
        mu_pred = vi_mu_logit 
        # 预测的方差，考虑温度衰减 (训练初期 T 大，允许探索；后期 T 小，分布变尖锐)
        # sigma_pred = (vi_log_sigma.exp() / max(temperature, 1e-6)).clamp(min=self.eps)
        sigma_pred = vi_log_sigma.exp().clamp(min=self.eps)  # 不使用温度衰减，直接使用预测的 sigma

        # ------------------------------------------------------------------
        # 2. 构造目标分布 T(Delta)
        # ------------------------------------------------------------------
        if vi_target is None:
            soft_labels = self._generate_soft_labels(dist_to_gt, pos_thresh) # thresh内部为1，外部逐渐衰减
        else:
            soft_labels = vi_target
            
        # 生成目标分布参数 (Logit 空间)
        mu_target, sigma_target = self._generate_target_gaussian_params(soft_labels)
        # mu_target 范围为 (-inf, +inf) 太大了，clamp到正负硬分配阈值的范围内
        mu_target = mu_target.clamp(min=-10.0, max=10.0)

        # ------------------------------------------------------------------
        # 3. Masking (构建有效监督区域)
        # ------------------------------------------------------------------        
        # A. 全局忽略点
        if is_ignored is not None:
            valid_global = (is_ignored == 0).float()
        else:
            valid_global = torch.ones_like(mu_pred)
            
        # B. 几何有效点 (去除过于极端的背景，防止 Logit -> -inf)
        # 只有 soft_label > 1e-3 的点才参与训练
        valid_geo = (soft_labels > 1e-3).float()
        
        valid_mask = valid_global * valid_geo
        num_valid = valid_mask.sum().clamp(min=1.0)
        
        # 3. Masking
        if is_pos is not None:
            pos_mask = is_pos.float()
            neg_mask = (1.0 - pos_mask) * (~is_ignored).float() if is_ignored is not None else (1.0 - pos_mask)
        else:
            pos_mask = (soft_labels > 0.5).float()
            neg_mask = (soft_labels <= 0.5).float()

        num_pos = pos_mask.sum().clamp(min=1.0)
        num_neg = neg_mask.sum().clamp(min=1.0)

        # ------------------------------------------------------------------
        # 4. 重构项 (Reconstruction Loss) - 仿照 VPD 的采样回归
        # ------------------------------------------------------------------
        # 重参数化采样: z = mu + sigma * eps
        eps = torch.randn_like(mu_pred)
        sampled_pred = mu_pred + sigma_pred * eps * 10
        
        # 在 Logit 空间计算 L1 Loss (比 MSE 更鲁棒，类似 Smooth L1)
        # 这迫使预测分布的采样值尽可能靠近目标 Logit
        # loss_recon = torch.abs(sampled_pred - mu_target)
        # or smoothl1
        loss_recon = F.smooth_l1_loss(sampled_pred, mu_target, reduction='none', beta=0.1)
        # loss_recon = (loss_recon * valid_mask).sum() / num_valid
        loss_recon_pos = (loss_recon * pos_mask).sum() / num_pos
        loss_recon_neg = (loss_recon * neg_mask).sum() / num_neg
        loss_recon = 0.5 * (loss_recon_pos + loss_recon_neg)

        # ------------------------------------------------------------------
        # 5. 正则项 (Distribution Matching) - 仿照 VPD 的 JS 散度
        # ------------------------------------------------------------------
        # 计算 KL 散度: KL(Pred || Target)
        # 强制预测分布的形状 (sigma) 去拟合目标分布的形状
        kl_val = self._js_divergence_gaussian(mu_pred, sigma_pred, mu_target, sigma_target)
        # loss_kl = (kl_val * valid_mask).sum() / num_valid
        loss_kl_pos = (kl_val * pos_mask).sum() / num_pos
        loss_kl_neg = (kl_val * neg_mask).sum() / num_neg
        loss_kl = 0.5 * (loss_kl_pos + loss_kl_neg)

        # ------------------------------------------------------------------
        # 6. 额外的 L2 正则 (可选，防止 sigma 异常)
        # ------------------------------------------------------------------
        loss_reg_sigma = (vi_log_sigma ** 2).mean()

        # ------------------------------------------------------------------
        # 7. 总 Loss
        # ------------------------------------------------------------------
        # VPD 风格通常直接相加或加权
        # 重构项保证位置准确，散度项保证不确定性合理
        loss_total = self.lambda_vi * (loss_recon + 0.01 * loss_kl + 0.0 * loss_reg_sigma) * 0.001
        
        loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=1e4, neginf=-1e4)

        # ------------------------------------------------------------------
        # 8. 输出推理概率
        # ------------------------------------------------------------------
        # 推理时，直接将预测的 Logit 转换为概率
        vi_prob = torch.sigmoid(mu_pred).detach()

        return dict(
            loss_total=loss_total,
            loss_vi_recon=loss_recon.detach(),
            loss_vi_kl=loss_kl.detach(),
            loss_vi_reg=loss_reg_sigma.detach(),
            vi_prob=vi_prob,  # already detached
            vi_sigma=sigma_pred.detach()
        )
