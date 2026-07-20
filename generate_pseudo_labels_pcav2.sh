GPU_NUM=4 GPU_IDS=0,1,2,3 START_STAGE=2 END_STAGE=3 \
CFG_STAGE1=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/train_cpm_vi_point_dotav10.py \
CFG_STAGE2=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/generate_pseudo_label_dotav10.py \
CFG_STAGE3=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/redet_dotav10.py \
NAME="0719_gaussian_sampled_tower_layered_dilated" \
ENABLE_FROZEN_V3_FINETUNE=False \
SAVE_VARIANCE_MAP=True \
RESUME_STAGE3=True \
DATA_ROOT=/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/ \
bash ./tools/run_vpd_full_pipeline.sh

generate_cpmvpd_point_pseudo_dotav10

PSEUDO_DIR=/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/trainval/annfiles/ \
PSEUDO_DIR=/media/ps/passport2/zlk/results/0525_xy_vpdstyle_bins21/vpd_cpm_dotav10/pseudo_labels_original \
STAGE1_CKPT=/media/ps/passport2/zlk/results/0527_xy_vpdstyle_cpmoriginal_clsw1_l5e-2/vpd_cpm_dotav10/epoch_6.pth \
CFG_STAGE2=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/generate_cpmvpd_seg_pseudo_dotav10.py \
CFG_STAGE2=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/generate_pseudo_label_dotav10.py \
CFG_STAGE2=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/generate_cpmvpd_fused_pseudo_dotav10.py \
CFG_STAGE1=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/train_cpm_vpd_point_dotav10.py \
