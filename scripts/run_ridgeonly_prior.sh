#!/bin/bash
# Fine-tune APENAS da camada ridge do subj01, a partir do checkpoint
# multi-sujeito (pre-treinado nos outros 7 sujeitos).
# Metrica: retrieval top-1 forward/backward no test set (calculada a cada epoca).
#
# Duas configuracoes:
#   padrao          hidden_dim 1024, sem low-level  -> ckpt multisubject_..._nolow
#   BLURRY=1        hidden_dim 4096, com low-level  -> ckpt final_multisubject_subj01
#                   (esta e a comparacao pareada com o paper)
set -e
set -o pipefail   # sem isso, falha do python fica mascarada pelo tee
source "$(dirname "$0")/common.sh"

MODEL_NAME=${MODEL_NAME:-subj01_ridgeonly_1sess_prior}

HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_SESSIONS=${NUM_SESSIONS:-1}
BLURRY=${BLURRY:-0}

if [ "$BLURRY" = "1" ]; then
    BLURRY_FLAGS="--blurry_recon --blur_scale=.5"
    MSCKPT=${MSCKPT:-$DATA/train_logs/final_multisubject_subj01}
    if [ "$HIDDEN_DIM" != "4096" ]; then
        echo "AVISO: BLURRY=1 exige o ckpt final_multisubject_subj01, que e 4096." >&2
        echo "       Com HIDDEN_DIM=$HIDDEN_DIM o load do ckpt vai falhar." >&2
    fi
    # Medido na A4500 (20 GB, desktop na mesma placa): o modelo 4096 tem 2,23 bi
    # de parametros e so cabe com os modulos congelados em fp16 (--frozen_fp16,
    # 8,30 GB -> 4,27 GB de pesos). Mesmo assim o pico chega a ~18,0 GB de
    # tensores, entao o batch cai de 16 para 8; com 16 o forward do prior estoura.
    FROZEN_FLAGS="--frozen_fp16"
    DEFAULT_BATCH=8
else
    BLURRY_FLAGS="--no-blurry_recon"
    MSCKPT=${MSCKPT:-$DATA/train_logs/multisubject_subj01_1024hid_nolow_300ep}
    FROZEN_FLAGS="--no-frozen_fp16"
    DEFAULT_BATCH=16
fi

# FROZEN=1 força o fp16 dos congelados também na config 1024. O default fica
# desligado lá para preservar a configuracao exata do braco de 1 sessao ja
# publicado, mas com 40 sessoes ele e necessario: em batch 16 o backward chega a
# 18,0 GB e basta o desktop abrir algumas janelas (~350 MB) para estourar. Na
# 1024, 713,3M dos 729,3M params estao congelados (97,8%), entao guardar em fp16
# libera ~1,4 GB sem mexer no batch -- preferivel a baixar o batch, que mudaria a
# dinamica de treino e a comparabilidade.
if [ "$FROZEN" = "1" ]; then FROZEN_FLAGS="--frozen_fp16"; fi
if [ "$FROZEN" = "0" ]; then FROZEN_FLAGS="--no-frozen_fp16"; fi

# PRIOR=0 treina sem o diffusion prior: a ridge so ve a loss contrastiva. O
# resto da configuracao fica igual, para comparar com e sem prior.
if [ "${PRIOR:-1}" = "0" ]; then PRIOR_FLAGS="--no-use_prior"
else PRIOR_FLAGS="--use_prior --prior_scale=30"; fi

# SEED muda a semente (padrao 42). Com a mesma semente e o mesmo codigo o treino
# e deterministico: repetir a corrida reproduz as metricas epoca por epoca.
#
# RESUME=1 continua do last.pth deste modelo, se existir, e anexa ao train.log
# em vez de sobrescrever. Combine com CKPT_INTERVAL=1 em corridas longas.
if [ "$RESUME" = "1" ]; then RESUME_FLAG="--resume"; TEE_FLAGS="-a"
else RESUME_FLAG="--no-resume"; TEE_FLAGS=""; fi

# ATENCAO: o Train.ipynb calcula batch_size a partir de GLOBAL_BATCH_SIZE, mas
# o argparse depois sobrescreve com o default de --batch_size (16). Por isso a
# variavel de ambiente sozinha NAO tem efeito -- e preciso passar --batch_size
# explicitamente, como o accel.slurm do repo original faz.
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$DEFAULT_BATCH}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH}   # 16 no 1024; 8 no 4096+blurry

# save_ckpt grava sempre em last.pth, entao um CKPT_INTERVAL menor custa um
# arquivo so e protege contra perder a corrida inteira (a maquina e resetada
# sem aviso -- ja aconteceu tres vezes).
mkdir -p $TRAIN_LOGS/$MODEL_NAME

$ENVP/bin/python train_ridgeonly.py \
    --data_path=$DATA \
    `# sd_image_var_autoenc e convnext ficam em $DATA, nao em $DATA/.cache` \
    --cache_dir=$DATA \
    --model_name=$MODEL_NAME \
    --multisubject_ckpt=$MSCKPT \
    --ridge_only \
    --no-multi_subject --subj=1 --num_sessions=$NUM_SESSIONS --batch_size=$BATCH_SIZE \
    --hidden_dim=$HIDDEN_DIM --n_blocks=4 \
    --clip_scale=1. $BLURRY_FLAGS $FROZEN_FLAGS $PRIOR_FLAGS \
    --max_lr=${MAX_LR:-3e-4} --mixup_pct=.33 --num_epochs=${NUM_EPOCHS:-150} \
    --no-use_image_aug --new_test --embedder_fp16 --seed=${SEED:-42} \
    --ckpt_interval=${CKPT_INTERVAL:-999} --ckpt_saving --no-wandb_log $RESUME_FLAG \
    --metrics_csv=$TRAIN_LOGS/$MODEL_NAME/metrics.csv \
    2>&1 | tee $TEE_FLAGS $TRAIN_LOGS/$MODEL_NAME/train.log
