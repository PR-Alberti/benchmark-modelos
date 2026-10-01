#!/bin/bash
# Gera as reconstrucoes de um modelo treinado: recon_inference + enhanced_recon_inference.
#
#   scripts/run_recon.sh <model_name> <hidden_dim> <blurry|noblurry> [--skip-enhanced]
#
# Exemplos:
#   scripts/run_recon.sh subj01_ridgeonly_1sess_prior 1024 noblurry
#   scripts/run_recon.sh final_subj01_pretrained_1sess_24bs 4096 blurry
#
# SEED muda a semente da amostragem (padrao 42) no recon e no refinamento.
#
# Etapa cuja saida ja existe e pulada; uma interrompida retoma do parcial
# (*_recon_partial.pt / *_enhanced_partial.pt). Para refazer, apague a saida.
set -e
set -o pipefail   # sem isso, falha do python fica mascarada pelo tee
source "$(dirname "$0")/common.sh"

MODEL_NAME="$1"
HIDDEN_DIM="${2:-1024}"
BLURRY_MODE="${3:-noblurry}"
SKIP_ENHANCED="${4:-}"

[ -z "$MODEL_NAME" ] && { sed -n '2,12p' "$SELF"; exit 1; }



if [ "$BLURRY_MODE" = "blurry" ]; then BLURRY_FLAG="--blurry_recon"; else BLURRY_FLAG="--no-blurry_recon"; fi
# FROZEN_CKPT=<ckpt de partida>/last.pth: backbone e prior vem de la, e do
# modelo so a ridge. Para os ridge-only treinados sem prior ou com
# --frozen_fp16 (veja --frozen_ckpt no recon_inference.py)
FROZEN_FLAG=""
if [ -n "$FROZEN_CKPT" ]; then FROZEN_FLAG="--frozen_ckpt=$FROZEN_CKPT"; fi
# o backbone 4096 em fp32 (7,6 GB) mais o unCLIP nao cabem nos 20 GB; em fp16 o
# resultado e o mesmo (ver --backbone_fp16 no recon_inference.py)
if [ "${BACKBONE_FP16:-$([ "$HIDDEN_DIM" = "4096" ] && echo 1 || echo 0)}" = "1" ]; then
    FP16_FLAG="--backbone_fp16"; else FP16_FLAG="--no-backbone_fp16"; fi

LOGDIR=$TRAIN_LOGS/$MODEL_NAME
mkdir -p "$LOGDIR"

OUT=$EVALS/$MODEL_NAME/$MODEL_NAME
# o clipvoxels e o ultimo tensor gravado; o parcial so e apagado depois dele
if [ -f "${OUT}_all_clipvoxels.pt" ] && [ ! -f "${OUT}_recon_partial.pt" ]; then
    echo "=== recon_inference: $MODEL_NAME ja feito, pulando ==="
else
    echo "=== recon_inference: $MODEL_NAME (hidden_dim=$HIDDEN_DIM, $BLURRY_MODE) ==="
    $ENVP/bin/python recon_inference.py \
        --model_name="$MODEL_NAME" \
        --data_path=$DATA --cache_dir=$DATA \
        --subj=1 --hidden_dim="$HIDDEN_DIM" --n_blocks=4 \
        $BLURRY_FLAG $FROZEN_FLAG $FP16_FLAG --new_test --seed="${SEED:-42}" \
        2>&1 | tee -a "$LOGDIR/recon.log"
fi

if [ "$SKIP_ENHANCED" = "--skip-enhanced" ]; then
    echo "=== enhanced pulado ==="; exit 0
fi

if [ -f "${OUT}_all_enhancedrecons.pt" ]; then
    echo "=== enhanced_recon_inference: $MODEL_NAME ja feito, pulando ==="
else
    echo "=== enhanced_recon_inference: $MODEL_NAME ==="
    $ENVP/bin/python enhanced_recon_inference.py \
        --model_name="$MODEL_NAME" \
        --data_path=$DATA --cache_dir=$DATA --subj=1 --seed="${SEED:-42}" \
        2>&1 | tee -a "$LOGDIR/enhanced.log"
fi

echo "=== pronto: $EVALS/$MODEL_NAME/ ==="
ls -la "$EVALS/$MODEL_NAME/"
