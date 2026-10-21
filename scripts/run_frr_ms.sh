#!/bin/bash
# FRR multi-sujeito (src/run_frr_ms.py): pre-treino linear de um espaco compartilhado de k dimensoes nos
# sujeitos 2-8 (modelo C) ou so no proprio subj01 (EXTRA=--pretreino=proprio, o controle de posto
# reduzido, modelo B), e ajuste fino so da ridge do subj01. Mesmo treino, teste e alvo do run_frr.sh.
#
#   NUM_SESSIONS=1 MODEL_NAME=subj01_frr_multisujeito_1sess scripts/run_frr_ms.sh
#   NUM_SESSIONS=1 MODEL_NAME=subj01_frr_postoreduzido_1sess EXTRA=--pretreino=proprio scripts/run_frr_ms.sh
#
# Precisa dos sujeitos 2-8 no disco (setup/bootstrap.sh --multisujeito, ~12 GB); os alvos CLIP deles
# (~50 GB) sao calculados na primeira corrida. O pre-treino dos outros sujeitos e um so para todos os
# modelos com a mesma configuracao (train_logs/_pretreino_frr_ms/) e retoma da ultima iteracao; o
# ajuste fino retoma da ultima dobra (train_logs/<modelo>/frr_cv). DATASET=<manifesto.json> treina o
# subj01 num dataset controlado, como no run_frr.sh.
set -e
set -o pipefail   # sem isso, falha do python fica mascarada pelo tee
source "$(dirname "$0")/common.sh"
prepara_dataset

NUM_SESSIONS=${NUM_SESSIONS:-1}
MODEL_NAME=${MODEL_NAME:-subj01_frr_multisujeito_${NUM_SESSIONS}sess}

mkdir -p "$TRAIN_LOGS/$MODEL_NAME"
echo "=== FRR multi-sujeito: $MODEL_NAME ($NUM_SESSIONS sessoes) ==="
$ENVP/bin/python run_frr_ms.py --model_name="$MODEL_NAME" --data_path="$DATA" --subj=1 \
    --num_sessions="$NUM_SESSIONS" ${DATASET:+--dataset="$DATASET"} $EXTRA \
    2>&1 | grep --line-buffered -v -E "Warning|pynvml|warnings.warn|pkg_resources" \
    | tee -a "$TRAIN_LOGS/$MODEL_NAME/frr_ms.log"
atualiza_site
