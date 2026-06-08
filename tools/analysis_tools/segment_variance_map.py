import torch


def _infer_image_shape(img_meta):
	for key in ('img_shape', 'pad_shape', 'ori_shape'):
		shape = img_meta.get(key, None)
		if shape is not None and len(shape) >= 2:
			return float(shape[0]), float(shape[1])
	raise KeyError('img_meta must contain one of img_shape/pad_shape/ori_shape.')


def _extract_center_and_size(gt_bboxes):
	if gt_bboxes is None or gt_bboxes.numel() == 0:
		return None, None, None, None

	boxes = gt_bboxes.float()
	if boxes.dim() == 1:
		boxes = boxes.unsqueeze(0)

	if boxes.size(1) >= 8:
		xs = boxes[:, 0:8:2]
		ys = boxes[:, 1:8:2]
		cx = xs.mean(dim=1)
		cy = ys.mean(dim=1)
		bw = (xs.max(dim=1)[0] - xs.min(dim=1)[0]).clamp(min=1e-3)
		bh = (ys.max(dim=1)[0] - ys.min(dim=1)[0]).clamp(min=1e-3)
	elif boxes.size(1) >= 5:
		cx = boxes[:, 0]
		cy = boxes[:, 1]
		bw = boxes[:, 2].abs().clamp(min=1e-3)
		bh = boxes[:, 3].abs().clamp(min=1e-3)
	elif boxes.size(1) >= 4:
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]
		cx = 0.5 * (x1 + x2)
		cy = 0.5 * (y1 + y2)
		bw = (x2 - x1).abs().clamp(min=1e-3)
		bh = (y2 - y1).abs().clamp(min=1e-3)
	else:
		raise ValueError(f'Unsupported gt bbox shape: {tuple(boxes.shape)}')

	return cx, cy, bw, bh


# def _get_gt_xy_knndist(gt_bboxes, img_meta, k=4):


def _gaussian_priors(h, w, cx, cy, sigma):
	yy, xx = torch.meshgrid(
		torch.arange(h, device=cx.device, dtype=torch.float32),
		torch.arange(w, device=cx.device, dtype=torch.float32),
		indexing='ij')
	dx2 = (xx.unsqueeze(0) - cx.view(-1, 1, 1))**2
	dy2 = (yy.unsqueeze(0) - cy.view(-1, 1, 1))**2
	denom = 2.0 * sigma.view(-1, 1, 1)**2 + 1e-6
	return torch.exp(-(dx2 + dy2) / denom)


def _compute_gt_guided_maps(p_model,
								gt_bboxes,
								img_meta,
								sigma_scale,
								min_sigma,
								max_sigma,
								score_thr,
								topk,
								bg_std_scale):
	p_model = torch.nan_to_num(p_model.float(), nan=0.0, posinf=1.0, neginf=0.0)
	if p_model.numel() == 0:
		return p_model, None

	h, w = p_model.shape
	if gt_bboxes is None or gt_bboxes.numel() == 0:
		return p_model, None

	cx, cy, bw, bh = _extract_center_and_size(gt_bboxes)  # each one is [n_gt]
	if cx is None:
		return p_model, None

	img_h, img_w = _infer_image_shape(img_meta)
	sx = float(w) / max(float(img_w), 1.0)
	sy = float(h) / max(float(img_h), 1.0)

	cx_f = cx * sx
	cy_f = cy * sy
	bw_f = bw * sx
	bh_f = bh * sy

	sigma = sigma_scale * torch.sqrt((bw_f * bh_f).clamp(min=1e-6))
	sigma = torch.clamp(sigma, min=min_sigma, max=max_sigma)

	priors = _gaussian_priors(h=h, w=w, cx=cx_f, cy=cy_f, sigma=sigma)
	priors_sum = priors.sum(dim=0, keepdim=True)
	priors_sum = torch.clamp(priors_sum, min=1e-8)
	ownership = priors / priors_sum

	per_obj = p_model.unsqueeze(0) * ownership

	if score_thr > 0.0:
		per_obj = per_obj * (per_obj >= score_thr).to(per_obj.dtype)

	if topk > 0:
		n_obj = per_obj.shape[0]
		flat = per_obj.view(n_obj, -1)
		k = min(int(topk), flat.shape[1])
		if k > 0:
			kth = torch.topk(flat, k, dim=1)[0][:, -1].view(n_obj, 1, 1)
			per_obj = per_obj * (per_obj >= kth).to(per_obj.dtype)

	fused = per_obj.sum(dim=0)

	if bg_std_scale is not None:
		mu = fused.mean()
		std = fused.std(unbiased=False)
		bg_thr = mu + float(bg_std_scale) * std
		fg_mask = fused >= bg_thr
		fused = fused * fg_mask.to(fused.dtype)
	else:
		fg_mask = fused > 0

	max_vals, max_ids = per_obj.max(dim=0)
	label_map = torch.full((h, w), -1, dtype=torch.long, device=p_model.device)
	valid = fg_mask & (max_vals > 0)
	label_map[valid] = max_ids[valid]
	

	return fused.clamp(min=0.0, max=1.0), label_map


def build_gt_guided_remap_map(p_model,
							  gt_bboxes,
							  img_meta,
							  sigma_scale=0.5,
							  min_sigma=1.0,
							  max_sigma=20.0,
							  score_thr=0.0,
							  topk=0,
							  bg_std_scale=0.5):
	"""GT-guided decomposition map.

	Steps:
	1) Build Gaussian prior q(x, g_i) for each GT with adaptive sigma_i.
	2) Compute ownership weights w_i(x) = q_i / sum_k q_k.
	3) Decompose model score map: P_i(x) = P_model(x) * w_i(x).
	4) Optional thresholding and top-k keep for each P_i.
	5) Keep pixels where sum_i P_i is significantly above background.
	"""
	fused, _ = _compute_gt_guided_maps(
		p_model=p_model,
		gt_bboxes=gt_bboxes,
		img_meta=img_meta,
		sigma_scale=sigma_scale,
		min_sigma=min_sigma,
		max_sigma=max_sigma,
		score_thr=score_thr,
		topk=topk,
		bg_std_scale=bg_std_scale)
	return fused


def build_gt_guided_segmentation_mask(p_model,
								gt_bboxes,
								img_meta,
								sigma_scale=0.5,
								min_sigma=1.0,
								max_sigma=20.0,
								score_thr=0.02,
								topk=0,
								bg_std_scale=0.5):
	"""Build per-pixel GT-instance label map.

	Returns:
		Tensor: [H, W] long, -1 for background, otherwise gt instance id.
	"""
	_, label_map = _compute_gt_guided_maps(
		p_model=p_model,
		gt_bboxes=gt_bboxes,
		img_meta=img_meta,
		sigma_scale=sigma_scale,
		min_sigma=min_sigma,
		max_sigma=max_sigma,
		score_thr=score_thr,
		topk=topk,
		bg_std_scale=bg_std_scale)
	if label_map is None:
		h, w = p_model.shape
		label_map = torch.full((h, w), -1, dtype=torch.long, device=p_model.device)
	return label_map


def _decode_obb_from_probmap(per_gt_map, fg_mask, stride, mask_min_pixels=6):
    """
    Decode an oriented bounding box from a per-GT-object probability map, 
    using a weighted covariance method similar to the CPM seg head target generation.
    Args:
        per_gt_map (Tensor): shape [H, W], probability map for a single GT object (non-zero only within the GT mask).
        fg_mask (Tensor): shape [H, W], boolean mask of valid foreground pixels (e.g. from GT-guided segmentation).
        stride (float): feature map stride in pixels, used to scale the output box.
        mask_min_pixels (int): minimum number of pixels required in the masked region to produce a valid box.
    Returns:
        Tensor: shape [5], (cx, cy, w, h, angle) of the decoded oriented bounding box in image pixel coordinates, or None if decoding fails.
    """
    
    mask = (per_gt_map > 0) & fg_mask  # gt_mask
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() < mask_min_pixels:
        return None

    weights = per_gt_map[ys, xs].float()
    if float(weights.sum().item()) <= 0:
        return None

    pts = torch.stack([xs.float(), ys.float()], dim=1)
    mean = (weights.view(-1, 1) * pts).sum(dim=0) / weights.sum()  # this should be replaced by gt center
    centered = pts - mean

    w = weights.view(-1, 1)
    cov = (centered * w).t().matmul(centered) / weights.sum()
    eigvals, eigvecs = torch.linalg.eigh(cov)
    major = eigvecs[:, 1]
    if float(eigvals[1].item()) < 1e-8:
        major = pts.new_tensor([1.0, 0.0])
    major = major / torch.clamp(torch.norm(major), min=1e-6)
    minor = torch.stack([-major[1], major[0]])

    proj_u = centered.matmul(major)
    proj_v = centered.matmul(minor)
    mean_u = (weights * proj_u).sum() / weights.sum()
    mean_v = (weights * proj_v).sum() / weights.sum()
    var_u = (weights * (proj_u - mean_u)**2).sum() / weights.sum()
    var_v = (weights * (proj_v - mean_v)**2).sum() / weights.sum()

    # Same heuristic as the seg head: map weighted variance to rectangle side length.
    width_feat = torch.clamp(torch.sqrt(torch.clamp(12.0 * var_u, min=0.0)), min=1.0)
    height_feat = torch.clamp(torch.sqrt(torch.clamp(12.0 * var_v, min=0.0)), min=1.0)
    center_feat = mean

    cx = center_feat[0] * float(stride)
    cy = center_feat[1] * float(stride)
    w = width_feat * float(stride)
    h = height_feat * float(stride)
    angle = torch.atan2(major[1], major[0])
    return torch.stack([cx, cy, w, h, angle], dim=0)
