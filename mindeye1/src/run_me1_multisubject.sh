#!/bin/bash
# MindEye1 model + MindEye2 protocol (commands mirror MindEyeV2 src/accel.slurm).
# Safe to re-run: finished stages are skipped, interrupted ones resume from last.pth.
# Usage: run_me1_multisubject.sh [subj] ; env: ROOT, PY, MAX_RESTARTS
set -u
SUBJ=${1:-1}
ROOT=${ROOT:-$HOME/mindeye1_multisubject}
PY=${PY:-$HOME/.conda/envs/mindeye1ms/bin/python}
MAX_RESTARTS=${MAX_RESTARTS:-5}
cd "$(dirname "$0")"
COMMON="--data_path=$ROOT/data/mindeyev2 --logs_path=$ROOT/train_logs --subj=$SUBJ --hidden_dim=1024 --n_blocks=4 \
  --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --prior_scale=30 --clip_scale=1 --num_sessions=40 --ckpt_interval=5"
PRE=me1_multisubject_excludingsubj0${SUBJ}_40sess
FT=me1_finetuned_subj0${SUBJ}_40sess

stage() { # name, extra args...
  local name=$1; shift
  if [ -f "$ROOT/train_logs/$name/final_retrieval.json" ]; then echo "[$(date)] $name already finished"; return 0; fi
  for attempt in $(seq 0 "$MAX_RESTARTS"); do
    echo "[$(date)] $name: attempt $attempt"
    "$PY" Train_MindEye_multisubject.py --model_name="$name" $COMMON "$@" && return 0
    echo "[$(date)] $name: exited with error"; sleep 60
  done
  echo "[$(date)] $name: giving up after $MAX_RESTARTS restarts"; return 1
}

stage "$PRE" --multi_subject --batch_size=21 || exit 1
stage "$FT" --no-multi_subject --batch_size=24 --multisubject_ckpt="$ROOT/train_logs/$PRE" || exit 1
echo "[$(date)] ALL DONE"
