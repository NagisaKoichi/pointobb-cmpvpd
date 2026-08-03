import os
import glob
import numpy as np
import multiprocessing as mp
from mmrotate.core import poly2obb_np, eval_rbbox_map
from mmrotate.datasets.dota import DOTADataset

def main():
    gt_dir = "/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/trainval/annfiles"
    # gt_dir = "/media/ps/passport2/zlk/results/0601_xy_vpdstyle_clsw1_l5e-2_lkl0p5_onlylr/vpd_cpm_dotav10/pseudo_labels_seg"
    pseudo_dir = "/media/ps/passport2/zlk/results/0803_score_pred_js_discret_lrzero_hard_eps/vpd_cpm_dotav10/pseudo_labels"
    # pseudo_dir = "/media/ps/passport2/zlk/results/0511_obboriginal/vpd_cpm_dotav10/pseudo_labels"
    # pseudo_dir = "/media/ps/passport2/zlk/results/0529_xy_vpdstyle_cpmoriginal_clsw1_l5e-2_lkl0p5/vpd_cpm_dotav10/pseudo_labels_legacy"
    # pseudo_dir = "/media/ps/passport2/zlk/results/0607_xy_vpdstyle_clsw1_l5e-2_lkl2_fpnfreeze/vpd_cpm_dotav10/pseudo_labels_seg"
    version = "le90"

    classes = DOTADataset.CLASSES
    cls2id = {c:i for i,c in enumerate(classes)}

    # 用 GT 建立评估样本顺序和 annotations
    ds = DOTADataset(
        ann_file=gt_dir,
        pipeline=[],
        version=version,
        difficulty=100,
        test_mode=True,
        filter_empty_gt=False
    )
    annotations = [ds.get_ann_info(i) for i in range(len(ds))]
    stems = [os.path.splitext(info["filename"])[0] for info in ds.data_infos]

    det_results = []
    for stem in stems:
        per_cls = [[] for _ in classes]
        p = os.path.join(pseudo_dir, stem + ".txt")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    try:
                        poly = np.array(list(map(float, parts[:8])), dtype=np.float32)
                        cls_name = parts[8]
                        if cls_name not in cls2id:
                            continue
                        x, y, w, h, a = poly2obb_np(poly, version)
                        score = 1.0  # 伪标签没有置信度时的近似做法
                        per_cls[cls2id[cls_name]].append([x, y, w, h, a, score])
                    except Exception:
                        continue

        per_cls_np = []
        for arr in per_cls:
            if len(arr) == 0:
                per_cls_np.append(np.zeros((0, 6), dtype=np.float32))
            else:
                per_cls_np.append(np.array(arr, dtype=np.float32))
        det_results.append(per_cls_np)
        
    mAP50, _ = eval_rbbox_map(
        det_results,
        annotations,
        iou_thr=0.5,
        dataset=classes,
        nproc=4
    )
    print({"pseudo_mAP50": float(mAP50)})
    
    if False:
        bins = 10    
        # draw a histogram
        iou_thrs = np.linspace(1/bins, 1.0, bins)
        mAPs = []
        for iou_thr in iou_thrs:
            mAP, _ = eval_rbbox_map(
                det_results,
                annotations,
                iou_thr=iou_thr,
                dataset=classes,
                nproc=4
            )
            mAPs.append(float(mAP))
        # mAP_bins = np.array(mAPs)  # cumulative -> bin_specific
        # mAP_bins = np.diff(np.concatenate(([0], mAP_bins)))  # 转换为每个 IoU 阈值对应的增量 mAP
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(iou_thrs, mAPs, marker='o')
        plt.title("mAP vs IoU Threshold for Pseudo Labels")
        plt.xlabel("IoU Threshold")
        plt.ylabel("mAP")
        plt.grid()
        plt.savefig("pseudo_label_map_curve.png")
        plt.show()
    
    


if __name__ == '__main__':
    mp.freeze_support()
    main()