GPU_NUM=4 GPU_IDS=0,1,2,3 PORT=29931 START_STAGE=1 END_STAGE=2 \
WORK_DIR_STAGE1=/media/ps/passport2/zlk/results/0430_var_v0_1_dotav10 \
PSEUDO_DIR=/media/ps/passport2/zlk/results/0430_var_v0_1_dotav10/pseudo_labels \
DATA_ROOT=/media/ps/passport2/zlk/datasets/DOTAv10_split_ss/ \
bash ./tools/run_vpd_full_pipeline.sh