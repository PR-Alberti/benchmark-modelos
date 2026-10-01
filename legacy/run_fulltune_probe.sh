#!/bin/bash
# Fine-tune APENAS da camada ridge do subj01, a partir do checkpoint
# multi-sujeito (pre-treinado nos outros 7 sujeitos), com 1 sessao de dados.
# Metrica: retrieval top-1 forward/backward no test set (calculada a cada epoca).
set -e
cd "$(dirname "$0")"

ENVP=/home/al.pedro.alberti/envs/fmri
DATA=/home/al.pedro.alberti/mindeyev2
MSCKPT=$DATA/train_logs/multisubject_subj01_1024hid_nolow_300ep
MODEL_NAME=${MODEL_NAME:-probe_fulltune}

# ATENCAO: o Train.ipynb calcula batch_size a partir de GLOBAL_BATCH_SIZE, mas
# o argparse depois sobrescreve com o default de --batch_size (16). Por isso a
# variavel de ambiente sozinha NAO tem efeito -- e preciso passar --batch_size
# explicitamente, como o accel.slurm do repo original faz.
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-16}   # 16 e o que cabe em 20 GB com o prior ligado
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=$DATA/.cache          # pesos do open_clip bigG vao para ca (~10GB)
export TRANSFORMERS_CACHE=$DATA/.cache

mkdir -p ../train_logs/$MODEL_NAME

$ENVP/bin/python Train_ridgeonly.py \
    --data_path=$DATA \
    --cache_dir=$DATA/.cache \
    --model_name=$MODEL_NAME \
    --multisubject_ckpt=$MSCKPT \
    --no-ridge_only \
    --no-multi_subject --subj=1 --num_sessions=1 --batch_size=$BATCH_SIZE \
    --hidden_dim=1024 --n_blocks=4 \
    --clip_scale=1. --no-blurry_recon --use_prior --prior_scale=30 \
    --max_lr=${MAX_LR:-3e-4} --mixup_pct=.33 --num_epochs=${NUM_EPOCHS:-150} \
    --no-use_image_aug --new_test --embedder_fp16 \
    --ckpt_interval=999 --ckpt_saving --no-wandb_log \
    --metrics_csv=../train_logs/$MODEL_NAME/metrics.csv \
    2>&1 | tee ../train_logs/$MODEL_NAME/train.log
