GPU_NUM=4 GPU_IDS=0,1,2,3 START_STAGE=3 END_STAGE=4 \
CFG_STAGE1=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/train_cpm_vpd_dotav10.py \
CFG_STAGE2=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/gen_vpd_improved_full.py \
CFG_STAGE3=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/redet_dotav10.py \
STAGE1_CKPT=/media/ps/passport2/zlk/results/frozen_vpd_v3/epoch_3.pth \
PSEUDO_DIR=/media/ps/passport2/zlk/results/0511_xy_savedckpt/vpd_cpm_dotav10/pseudo_labels_merged/pseudo_labels \
NAME="0511_xy_savedckpt" \
ENABLE_FROZEN_V3_FINETUNE=False \
SAVE_VARIANCE_MAP=True \
RESUME_STAGE3=True \
DATA_ROOT=/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/ \
bash ./tools/run_vpd_full_pipeline.sh
