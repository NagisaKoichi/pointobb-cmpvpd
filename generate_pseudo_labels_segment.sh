GPU_NUM=4 GPU_IDS=0,1,2,3 START_STAGE=2 END_STAGE=4 \
CFG_STAGE2=/media/ps/passport2/zlk/PointOBB-v2-vpd/configs/pointobbv2/generate_cpmvpd_seg_pseudo_dotav10.py \
STAGE1_CKPT=/media/ps/passport2/zlk/results/frozen_vpd_v3/epoch_3.pth \
WORK_DIR_STAGE1=/media/ps/passport2/zlk/results/frozen_vpd_v3 \
PSEUDO_DIR=/media/ps/passport2/zlk/results/frozen_vpd_v3/pseudo_labels_seg_ep3 \
DATA_ROOT=/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/ \
bash ./tools/run_vpd_full_pipeline.sh