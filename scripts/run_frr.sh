#!/bin/bash
# FRR (fractional ridge regression) de voxels para o embedding CLIP, subj01, nas
# condicoes do benchmark: mesmas sessoes de treino, mesmo teste, mesmo alvo CLIP.
# Sem GPU a CV leva horas; com a A4500, 1 sessao leva ~1 min e 40 sessoes ~1 h.
#
#   scripts/run_frr.sh                        # 1 sessao   -> results/evals/subj01_frr_1sess
#   NUM_SESSIONS=40 scripts/run_frr.sh        # 40 sessoes -> results/evals/subj01_frr_40sess
#
# MODEL_NAME muda o nome da saida; EXTRA repassa opcoes ao run_frr.py, por exemplo
#   MODEL_NAME=subj01_frr_1sess_global EXTRA=--global_fraction scripts/run_frr.sh
#
# As dobras da validacao cruzada ficam em train_logs/<modelo>/frr_cv, e uma corrida
# interrompida retoma da primeira que falta. Os embeddings CLIP do treino ficam em
# cache em <dados>/clip_targets (8 GB para as 40 sessoes).
set -e
set -o pipefail   # sem isso, falha do python fica mascarada pelo tee
source "$(dirname "$0")/common.sh"

NUM_SESSIONS=${NUM_SESSIONS:-1}
MODEL_NAME=${MODEL_NAME:-subj01_frr_${NUM_SESSIONS}sess}


mkdir -p "$TRAIN_LOGS/$MODEL_NAME"
echo "=== FRR: $MODEL_NAME ($NUM_SESSIONS sessoes) ==="
$ENVP/bin/python run_frr.py --model_name="$MODEL_NAME" --data_path="$DATA" --subj=1 \
    --num_sessions="$NUM_SESSIONS" $EXTRA \
    2>&1 | grep --line-buffered -v -E "Warning|pynvml|warnings.warn|pkg_resources" \
    | tee -a "$TRAIN_LOGS/$MODEL_NAME/frr.log"
