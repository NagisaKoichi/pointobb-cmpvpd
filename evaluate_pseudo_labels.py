import os
import glob
import numpy as np
import multiprocessing as mp
from mmrotate.core import poly2obb_np, eval_rbbox_map
from mmrotate.datasets.dota import DOTADataset

def main():
    # 改成你的路径
    gt_dir = "/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/trainval/annfiles"
    # pseudo_dir = "/media/ps/passport2/zlk/results/0527_xy_vpdstyle_cpmoriginal/vpd_cpm_dotav10/pseudo_labels"
    pseudo_dir = "/media/ps/passport2/zlk/results/0511_obboriginal/vpd_cpm_dotav10/pseudo_labels"
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


if __name__ == '__main__':
    mp.freeze_support()
    main()