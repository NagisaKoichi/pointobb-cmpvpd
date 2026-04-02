#!/usr/bin/env bash
set -euo pipefail

# One-click 4-stage pipeline for PointOBB-v2-vpd:
#   1) Train VPD-CPM
#   2) Generate pseudo labels
#   3) Train detector (ReDet)
#   4) Evaluate and/or export submission
#
# Usage examples:
#   GPU_NUM=1 GPU_IDS=0 ./tools/run_vpd_full_pipeline.sh
#   GPU_NUM=4 GPU_IDS=0,1,2,3 START_STAGE=2 END_STAGE=4 ./tools/run_vpd_full_pipeline.sh
#   SAVE_SUBMISSION=True SUBMISSION_DIR=/abs/path/submission ./tools/run_vpd_full_pipeline.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -----------------------------
# User-configurable parameters
# -----------------------------
DATA_ROOT="${DATA_ROOT:-/media/passport2/zlk/datasets/DOTAv10_split_ss/}"

# Stage-1 VPD config (point-supervised by default)
CFG_STAGE1="${CFG_STAGE1:-${REPO_ROOT}/configs/pointobbv2/train_cpm_vpd_point_dotav10.py}"
# You may switch to full-supervision VPD with:
# CFG_STAGE1=${REPO_ROOT}/configs/pointobbv2/train_cpm_vpd_dotav10.py

CFG_STAGE2="${CFG_STAGE2:-${REPO_ROOT}/configs/pointobbv2/generate_pseudo_label_dotav10.py}"
CFG_STAGE3="${CFG_STAGE3:-${REPO_ROOT}/configs/pointobbv2/redet_dotav10.py}"

WORK_DIR_STAGE1="${WORK_DIR_STAGE1:-${REPO_ROOT}/work_dirs/vpd_cpm_dotav10}"
WORK_DIR_STAGE3="${WORK_DIR_STAGE3:-${REPO_ROOT}/work_dirs/redet_dotav10_vpd}"

# Stage-2 pseudo label directory.
# Keep trailing slash for writer path because code concatenates file names directly.
PSEUDO_DIR="${PSEUDO_DIR:-${WORK_DIR_STAGE1}/pseudo_labels}"
PSEUDO_DIR_WRITE="${PSEUDO_DIR}/"

TEST_WORK_DIR="${TEST_WORK_DIR:-${WORK_DIR_STAGE3}/test_eval}"
SUBMISSION_DIR="${SUBMISSION_DIR:-${WORK_DIR_STAGE3}/submission}"

TRAIN_ANN="${TRAIN_ANN:-${DATA_ROOT}/trainval/annfiles/}"
TRAIN_IMG="${TRAIN_IMG:-${DATA_ROOT}/trainval/images/}"
TEST_IMG="${TEST_IMG:-${DATA_ROOT}/test/images/}"
METRIC_ANN="${METRIC_ANN:-${TRAIN_ANN}}"
METRIC_IMG="${METRIC_IMG:-${TRAIN_IMG}}"

GPU_NUM="${GPU_NUM:-1}"
GPU_IDS="${GPU_IDS:-0}"
PORT="${PORT:-29801}"
TEST_PORT="${TEST_PORT:-29816}"

START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"

RESUME_STAGE1="${RESUME_STAGE1:-True}"
RESUME_STAGE3="${RESUME_STAGE3:-False}"

# "auto" means resolve latest epoch checkpoint in corresponding work_dir.
STAGE1_CKPT="${STAGE1_CKPT:-auto}"
DET_CKPT="${DET_CKPT:-auto}"

PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-auto}"
CALC_METRICS="${CALC_METRICS:-True}"
SAVE_SUBMISSION="${SAVE_SUBMISSION:-False}"
CREATE_SYMLINK="${CREATE_SYMLINK:-False}"

resolve_latest_epoch_ckpt() {
  local work_dir="$1"
  ls -1v "$work_dir"/epoch_*.pth 2>/dev/null | tail -n 1 || true
}

resolve_submission_dir() {
  local requested="$1"
  if [[ -e "$requested" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    echo "${requested}_${ts}"
  else
    echo "$requested"
  fi
}

resolve_pretrained_backbone() {
  local default_rel="${REPO_ROOT}/work_dirs/pretrain_model/re_resnet50_c8_batch256-25b16846.pth"
  local parent_abs
  parent_abs="$(cd "${REPO_ROOT}/.." && pwd)/re_resnet50_c8_batch256-25b16846.pth"
  local fixed_abs="/media/passport2/zlk/re_resnet50_c8_batch256-25b16846.pth"

  if [[ "$PRETRAINED_BACKBONE" == "auto" ]]; then
    if [[ -f "$default_rel" ]]; then
      PRETRAINED_BACKBONE="$default_rel"
    elif [[ -f "$parent_abs" ]]; then
      PRETRAINED_BACKBONE="$parent_abs"
    elif [[ -f "$fixed_abs" ]]; then
      PRETRAINED_BACKBONE="$fixed_abs"
    else
      PRETRAINED_BACKBONE="None"
    fi
  fi
}

run_single_gpu() {
  local cfg="$1"
  local work_dir="$2"
  shift 2
  python "${REPO_ROOT}/tools/train.py" "$cfg" --work-dir "$work_dir" --gpu-ids "$GPU_IDS" "$@"
}

run_dist() {
  local cfg="$1"
  local work_dir="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$PORT" "${REPO_ROOT}/tools/dist_train.sh" "$cfg" "$GPU_NUM" --work-dir "$work_dir" "$@"
}

run_dist_resume() {
  local cfg="$1"
  local ckpt="$2"
  local work_dir="$3"
  shift 3
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$PORT" "${REPO_ROOT}/tools/dist_train_resume.sh" "$cfg" "$ckpt" "$GPU_NUM" --work-dir "$work_dir" "$@"
}

run_eval_single_gpu() {
  local cfg="$1"
  local ckpt="$2"
  python "${REPO_ROOT}/tools/test.py" "$cfg" "$ckpt" \
    --gpu-ids "$GPU_IDS" \
    --work-dir "$TEST_WORK_DIR" \
    --eval mAP \
    --cfg-options data.test.ann_file="${METRIC_ANN}" data.test.img_prefix="${METRIC_IMG}" \
    model.backbone.pretrained=${PRETRAINED_BACKBONE} \
    --eval-options metric=mAP \
    --eval-options iou_thr=0.5
}

run_eval_dist() {
  local cfg="$1"
  local ckpt="$2"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$TEST_PORT" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python -m torch.distributed.launch \
      --nproc_per_node="$GPU_NUM" \
      --master_port="$TEST_PORT" \
      "${REPO_ROOT}/tools/test.py" \
      "$cfg" "$ckpt" \
      --launcher pytorch \
      --work-dir "$TEST_WORK_DIR" \
      --eval mAP \
      --cfg-options data.test.ann_file="${METRIC_ANN}" data.test.img_prefix="${METRIC_IMG}" \
      model.backbone.pretrained=${PRETRAINED_BACKBONE} \
      --eval-options metric=mAP \
      --eval-options iou_thr=0.5
}

run_submission_single_gpu() {
  local cfg="$1"
  local ckpt="$2"
  python "${REPO_ROOT}/tools/test.py" "$cfg" "$ckpt" \
    --gpu-ids "$GPU_IDS" \
    --work-dir "$TEST_WORK_DIR" \
    --format-only \
    --cfg-options data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
    model.backbone.pretrained=${PRETRAINED_BACKBONE} \
    --eval-options submission_dir="$SUBMISSION_DIR"
}

run_submission_dist() {
  local cfg="$1"
  local ckpt="$2"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PORT="$TEST_PORT" PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python -m torch.distributed.launch \
      --nproc_per_node="$GPU_NUM" \
      --master_port="$TEST_PORT" \
      "${REPO_ROOT}/tools/test.py" \
      "$cfg" "$ckpt" \
      --launcher pytorch \
      --work-dir "$TEST_WORK_DIR" \
      --format-only \
      --cfg-options data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
      model.backbone.pretrained=${PRETRAINED_BACKBONE} \
      --eval-options submission_dir="$SUBMISSION_DIR"
}

print_output_layout() {
  cat <<EOF
==================== Output Layout ====================
[Stage1] VPD-CPM work_dir:
  ${WORK_DIR_STAGE1}
  - checkpoints: ${WORK_DIR_STAGE1}/epoch_*.pth
  - latest checkpoint symlink (optional): ${WORK_DIR_STAGE1}/latest.pth
  - train logs: ${WORK_DIR_STAGE1}/*.log, ${WORK_DIR_STAGE1}/*.log.json

[Stage2] Pseudo labels:
  ${PSEUDO_DIR}

[Stage3] Detector work_dir:
  ${WORK_DIR_STAGE3}
  - checkpoints: ${WORK_DIR_STAGE3}/epoch_*.pth
  - latest checkpoint symlink (optional): ${WORK_DIR_STAGE3}/latest.pth
  - train logs: ${WORK_DIR_STAGE3}/*.log, ${WORK_DIR_STAGE3}/*.log.json

[Stage4] Evaluation / Submission:
  - eval artifacts: ${TEST_WORK_DIR}
  - submission files: ${SUBMISSION_DIR}
=======================================================
EOF
}

mkdir -p "$WORK_DIR_STAGE1" "$WORK_DIR_STAGE3" "$PSEUDO_DIR" "$TEST_WORK_DIR"
resolve_pretrained_backbone

print_output_layout

echo "Using PRETRAINED_BACKBONE=${PRETRAINED_BACKBONE}"
echo "CALC_METRICS=${CALC_METRICS}, SAVE_SUBMISSION=${SAVE_SUBMISSION}"

if [[ "$START_STAGE" -le 1 && "$END_STAGE" -ge 1 ]]; then
  echo "[1/4] Train VPD-CPM"
  STAGE1_ARGS=()
  if [[ "$RESUME_STAGE1" == "True" ]]; then
    LATEST_STAGE1_CKPT="$(resolve_latest_epoch_ckpt "$WORK_DIR_STAGE1")"
    if [[ -n "$LATEST_STAGE1_CKPT" ]]; then
      echo "Resume stage-1 from: $LATEST_STAGE1_CKPT"
      STAGE1_ARGS+=(--resume-from "$LATEST_STAGE1_CKPT")
    fi
  fi

  if [[ "$GPU_NUM" -le 1 ]]; then
    run_single_gpu "$CFG_STAGE1" "$WORK_DIR_STAGE1" \
      "${STAGE1_ARGS[@]}" \
      --cfg-options data.train.ann_file="${TRAIN_ANN}" data.train.img_prefix="${TRAIN_IMG}" \
      data.val.ann_file="${TRAIN_ANN}" data.val.img_prefix="${TRAIN_IMG}" \
      data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
      model.train_cfg.store_dir="${WORK_DIR_STAGE1}" model.test_cfg.store_dir="${WORK_DIR_STAGE1}" \
      checkpoint_config.create_symlink=${CREATE_SYMLINK}
  else
    run_dist "$CFG_STAGE1" "$WORK_DIR_STAGE1" \
      "${STAGE1_ARGS[@]}" \
      --cfg-options data.train.ann_file="${TRAIN_ANN}" data.train.img_prefix="${TRAIN_IMG}" \
      data.val.ann_file="${TRAIN_ANN}" data.val.img_prefix="${TRAIN_IMG}" \
      data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
      model.train_cfg.store_dir="${WORK_DIR_STAGE1}" model.test_cfg.store_dir="${WORK_DIR_STAGE1}" \
      checkpoint_config.create_symlink=${CREATE_SYMLINK}
  fi
else
  echo "[1/4] Skip Stage 1"
fi

if [[ "$START_STAGE" -le 2 && "$END_STAGE" -ge 2 ]]; then
  echo "[2/4] Generate Pseudo Labels"
  if [[ "$STAGE1_CKPT" == "auto" ]]; then
    STAGE1_CKPT="$(resolve_latest_epoch_ckpt "$WORK_DIR_STAGE1")"
  fi

  if [[ -z "$STAGE1_CKPT" || ! -f "$STAGE1_CKPT" ]]; then
    echo "Stage-1 checkpoint not found: $STAGE1_CKPT" >&2
    exit 1
  fi

  if [[ "$GPU_NUM" -le 1 ]]; then
    run_single_gpu "$CFG_STAGE2" "$WORK_DIR_STAGE1" \
      --resume-from "$STAGE1_CKPT" \
      --cfg-options data.train.ann_file="${TRAIN_ANN}" data.train.img_prefix="${TRAIN_IMG}" \
      data.val.ann_file="${TRAIN_ANN}" data.val.img_prefix="${TRAIN_IMG}" \
      data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
      model.train_cfg.store_dir="${WORK_DIR_STAGE1}" model.train_cfg.store_ann_dir="${PSEUDO_DIR_WRITE}" \
      checkpoint_config.create_symlink=${CREATE_SYMLINK}
  else
    run_dist_resume "$CFG_STAGE2" "$STAGE1_CKPT" "$WORK_DIR_STAGE1" \
      --cfg-options data.train.ann_file="${TRAIN_ANN}" data.train.img_prefix="${TRAIN_IMG}" \
      data.val.ann_file="${TRAIN_ANN}" data.val.img_prefix="${TRAIN_IMG}" \
      data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
      model.train_cfg.store_dir="${WORK_DIR_STAGE1}" model.train_cfg.store_ann_dir="${PSEUDO_DIR_WRITE}" \
      checkpoint_config.create_symlink=${CREATE_SYMLINK}
  fi
else
  echo "[2/4] Skip Stage 2"
fi

if [[ "$START_STAGE" -le 3 && "$END_STAGE" -ge 3 ]]; then
  echo "[3/4] Train Detector (ReDet)"
  STAGE3_ARGS=()
  if [[ "$RESUME_STAGE3" == "True" ]]; then
    LATEST_DET_CKPT="$(resolve_latest_epoch_ckpt "$WORK_DIR_STAGE3")"
    if [[ -n "$LATEST_DET_CKPT" ]]; then
      echo "Resume detector from: $LATEST_DET_CKPT"
      STAGE3_ARGS+=(--resume-from "$LATEST_DET_CKPT")
    fi
  fi

  if [[ "$GPU_NUM" -le 1 ]]; then
    run_single_gpu "$CFG_STAGE3" "$WORK_DIR_STAGE3" \
      "${STAGE3_ARGS[@]}" \
      --cfg-options data.train.ann_file="${PSEUDO_DIR}" data.train.img_prefix="${TRAIN_IMG}" \
      data.val.ann_file="${TRAIN_ANN}" data.val.img_prefix="${TRAIN_IMG}" \
      data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
      model.backbone.pretrained=${PRETRAINED_BACKBONE} \
      checkpoint_config.create_symlink=${CREATE_SYMLINK}
  else
    run_dist "$CFG_STAGE3" "$WORK_DIR_STAGE3" \
      "${STAGE3_ARGS[@]}" \
      --cfg-options data.train.ann_file="${PSEUDO_DIR}" data.train.img_prefix="${TRAIN_IMG}" \
      data.val.ann_file="${TRAIN_ANN}" data.val.img_prefix="${TRAIN_IMG}" \
      data.test.ann_file="${TEST_IMG}" data.test.img_prefix="${TEST_IMG}" \
      model.backbone.pretrained=${PRETRAINED_BACKBONE} \
      checkpoint_config.create_symlink=${CREATE_SYMLINK}
  fi
else
  echo "[3/4] Skip Stage 3"
fi

if [[ "$START_STAGE" -le 4 && "$END_STAGE" -ge 4 ]]; then
  echo "[4/4] Test Detector"

  if [[ "$CALC_METRICS" != "True" && "$SAVE_SUBMISSION" != "True" ]]; then
    echo "Both CALC_METRICS and SAVE_SUBMISSION are False. Nothing to do in stage 4." >&2
    exit 1
  fi

  if [[ "$DET_CKPT" == "auto" ]]; then
    DET_CKPT="$(resolve_latest_epoch_ckpt "$WORK_DIR_STAGE3")"
  fi

  if [[ -z "$DET_CKPT" || ! -f "$DET_CKPT" ]]; then
    echo "Detector checkpoint not found: $DET_CKPT" >&2
    echo "Set it manually, e.g. DET_CKPT=${WORK_DIR_STAGE3}/epoch_12.pth" >&2
    exit 1
  fi

  mkdir -p "$TEST_WORK_DIR"

  if [[ "$CALC_METRICS" == "True" ]]; then
    echo "Stage4: evaluate mAP"
    if [[ "$GPU_NUM" -le 1 ]]; then
      run_eval_single_gpu "$CFG_STAGE3" "$DET_CKPT"
    else
      run_eval_dist "$CFG_STAGE3" "$DET_CKPT"
    fi
  fi

  if [[ "$SAVE_SUBMISSION" == "True" ]]; then
    SUBMISSION_DIR="$(resolve_submission_dir "$SUBMISSION_DIR")"
    echo "Using SUBMISSION_DIR=${SUBMISSION_DIR}"
    if [[ "$GPU_NUM" -le 1 ]]; then
      run_submission_single_gpu "$CFG_STAGE3" "$DET_CKPT"
    else
      run_submission_dist "$CFG_STAGE3" "$DET_CKPT"
    fi
  fi
else
  echo "[4/4] Skip Stage 4"
fi

echo "Done. submission_dir=${SUBMISSION_DIR}"
